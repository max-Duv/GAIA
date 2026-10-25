import os
import sys
import time
import subprocess
import csv
import argparse
import numpy as np
from scipy.stats import ttest_rel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "T1"))
sys.path.insert(0, os.path.join(PARENT_DIR, "T2"))

_EXE = ".exe" if sys.platform == "win32" else ""

from gpu_fault_assessor_v2 import GPUFaultAssessorV2
from gpu_defense_filters_v2 import PlatformAgnosticAnomalyDetector


def _find_jetson_devfreq():
    """Detect Jetson GPU devfreq node for clock capping."""
    for node in ['gpu-gpc-0']:
        path = f'/sys/class/devfreq/{node}'
        if os.path.isdir(path):
            return path
    return None


def _read_devfreq_max(devfreq_path):
    with open(os.path.join(devfreq_path, 'max_freq')) as f:
        return int(f.read().strip())


def _read_devfreq_available(devfreq_path):
    with open(os.path.join(devfreq_path, 'available_frequencies')) as f:
        return sorted(int(x) for x in f.read().split())


def _snap_to_available(available, target_hz):
    return min(available, key=lambda f: abs(f - target_hz))


def _set_devfreq_max(devfreq_path, freq_hz):
    freq_path = os.path.join(devfreq_path, 'max_freq')
    cmd = ['sudo', 'sh', '-c', f'echo {int(freq_hz)} > {freq_path}']
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"    [!] WARNING: Could not set GPU clock to {freq_hz/1e6:.0f} MHz: {res.stderr.strip()}")
        return False
    print(f"    [+] GPU max clock set to {freq_hz/1e6:.0f} MHz")
    return True


def get_current_power_limit(gpu_idx=0):
    """Reads the current enforced power limit (watts).
    Tries nvidia-smi query, then pynvml, then power.max_limit as fallback."""
    for field in ["power.limit", "power.max_limit"]:
        cmd = [
            "nvidia-smi", f"--id={gpu_idx}",
            f"--query-gpu={field}", "--format=csv,noheader,nounits"
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                val = res.stdout.strip()
                if val and val != "[N/A]" and val != "N/A":
                    return float(val)
        except Exception:
            pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        return limit_mw / 1000.0
    except Exception:
        pass

    print("    [!] WARNING: Cannot query power limit. Using 130W default (Jetson Thor TDP).")
    return 130.0


def set_power_limit(watts, gpu_idx=0):
    """Sets GPU power limit via nvidia-smi. Requires root/sudo on Linux.
    Returns True on success, False on failure (non-fatal)."""
    cmd = ["nvidia-smi", f"--id={gpu_idx}", f"-pl", str(int(watts))]
    if sys.platform != "win32":
        cmd.insert(0, "sudo")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"    [!] WARNING: Could not set power limit to {watts}W: {res.stderr.strip()}")
        print(f"    [!] T3 will still collect data but clamping may not take effect.")
        return False
    print(f"    [+] Power limit set to {int(watts)}W")
    return True


def _run_single_trial(
    trial_num,
    total_trials,
    assessor,
    detector,
    cuda_binary,
    stress_cmd,
    phases,
    duration_per_phase,
    apply_mitigation,
    restore_normal,
    mitigation_label,
):
    """Runs one complete pass through all phases. Returns (all_records, summary_results).
    apply_mitigation() and restore_normal() are callbacks for the platform-specific
    power/clock capping mechanism."""
    print(f"\n{'='*69}")
    print(f"  Trial {trial_num}/{total_trials}")
    print(f"{'='*69}")

    all_records = []
    summary_results = {}
    global_start = time.time()
    prev_current_a = 0.0
    prev_time = global_start

    for p_info in phases:
        p_name = p_info["name"]
        print(f"\n[+] Phase: '{p_name}' ...")

        detector.power_buffer.clear()
        detector.voltage_buffer.clear()
        detector.time_buffer.clear()

        cuda_proc = None
        mitigation_active = False
        mitigation_trigger_count = 0

        if p_info.get("force_clamp"):
            print(f"    [+] Applying clamped {mitigation_label}")
            apply_mitigation()

        if p_info["run_stress"]:
            try:
                cuda_proc = subprocess.Popen(
                    stress_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                print(f"    [+] Stress process launched (PID: {cuda_proc.pid})")
                time.sleep(0.5)
            except Exception as e:
                raise RuntimeError(f"Failed to launch stress process: {e}")

        latencies_tot = []
        latencies_compute = []
        phase_start = time.time()

        while (time.time() - phase_start) < duration_per_phase:
            v_res = assessor.run_victim_workload_iteration()
            latencies_tot.append(v_res["total_latency_ms"])
            latencies_compute.append(v_res["gpu_compute_ms"])

            t_curr = time.time()
            dt = max(t_curr - prev_time, 0.001)
            hw = assessor.get_hardware_telemetry()

            p_w = hw["power_w"]
            v_meas = hw.get("voltage_v", 0.0)
            if v_meas <= 0:
                v_meas = 19.75
            curr_a = p_w / v_meas
            di_dt = (curr_a - prev_current_a) / dt

            detector.update_buffers(t_curr - global_start, p_w, v_meas)
            spec_result = detector.analyze_power_level()
            is_anomalous = spec_result[0]
            mean_power_det = spec_result[1]

            if p_info["mitigate"] and is_anomalous and not mitigation_active:
                print(f"    [!] Anomaly detected (mean power: {mean_power_det:.1f} W > {detector.power_anomaly_threshold_w:.0f} W) — applying {mitigation_label}")
                clamp_ok = apply_mitigation()
                if clamp_ok:
                    mitigation_active = True
                    mitigation_trigger_count += 1
                else:
                    print(f"    [!] Mitigation clamp FAILED")

            if p_info.get("terminate") and is_anomalous and not mitigation_active:
                print(f"    [!] Anomaly detected (mean power: {mean_power_det:.1f} W > {detector.power_anomaly_threshold_w:.0f} W) — TERMINATING adversarial kernel")
                if cuda_proc is not None and cuda_proc.poll() is None:
                    cuda_proc.terminate()
                    try:
                        cuda_proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        cuda_proc.kill()
                    print(f"    [+] Adversarial kernel terminated (PID: {cuda_proc.pid})")
                    cuda_proc = None
                mitigation_active = True
                mitigation_trigger_count += 1

            all_records.append({
                "trial": trial_num,
                "timestamp": round(t_curr - global_start, 4),
                "phase": p_name,
                "data_source": hw["provenance"],
                "power_w": round(p_w, 3),
                "voltage_v": round(v_meas, 5),
                "current_a": round(curr_a, 4),
                "di_dt_amps_per_sec": round(di_dt, 4),
                "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
                "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                "anomaly_detected": int(is_anomalous),
                "mitigation_active": int(mitigation_active),
                "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                "gpu_temp_c": hw.get("gpu_temp_c", 0),
                "gpu_util_pct": hw.get("gpu_util_pct", 0),
                "mem_util_pct": hw.get("mem_util_pct", 0),
                "sw_power_cap": hw.get("sw_power_cap", 0),
                "hw_slowdown": hw.get("hw_slowdown", 0),
                "throttle_bitmask": hw.get("throttle_bitmask", 0)
            })

            prev_current_a = curr_a
            prev_time = t_curr
            time.sleep(0.02)

        if cuda_proc is not None:
            cuda_proc.terminate()
            try:
                cuda_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cuda_proc.kill()

        if mitigation_active or p_info.get("force_clamp"):
            print(f"    [+] Restoring normal operation")
            restore_normal()

        phase_records = [r for r in all_records if r["phase"] == p_name]
        mean_tot = float(np.mean(latencies_tot))
        mean_comp = float(np.mean(latencies_compute))
        mean_pow = float(np.mean([r["power_w"] for r in phase_records]))
        min_v = float(np.min([r["voltage_v"] for r in phase_records]))

        summary_results[p_name] = {
            "mean_total_latency_ms": mean_tot,
            "mean_gpu_compute_ms": mean_comp,
            "mean_power_w": mean_pow,
            "min_voltage_v": min_v,
            "mitigation_triggers": mitigation_trigger_count
        }

        print(f"    Phase '{p_name}' Completed:")
        print(f"      - Total Latency  : {mean_tot:.2f} ms (Compute: {mean_comp:.2f} ms)")
        print(f"      - Mean Power     : {mean_pow:.2f} W")
        print(f"      - Min Voltage    : {min_v:.4f} V")
        if p_info["mitigate"]:
            print(f"      - Mitigation Triggers: {mitigation_trigger_count}")

    return all_records, summary_results


def run_closed_loop_experiment(
    cuda_binary=None,
    duration_per_phase=15.0,
    mitigation_power_cap_w=None,
    clock_cap_mhz=None,
    detection_threshold_w=60.0,
    output_csv=None,
    gpu_idx=0,
    num_trials=5
):
    if cuda_binary is None:
        cuda_binary = os.path.join(PARENT_DIR, "T1", "gpu_power_poc_v3" + _EXE)
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "closed_loop_mitigation_v2.csv")

    if not os.path.exists(cuda_binary):
        raise FileNotFoundError(
            f"CUDA stress binary not found: {cuda_binary}\n"
            f"Run T1 first to compile it: python controlled_3phase_runner_v2.py --duration 5"
        )

    print("=====================================================================")
    print(" Project GAIA: T3 Closed-Loop Active Mitigation Experiment (v2.0)    ")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s")
    print("=====================================================================")

    assessor = GPUFaultAssessorV2(target_gpu_index=gpu_idx)

    devfreq_path = _find_jetson_devfreq()
    if devfreq_path:
        available_freqs = _read_devfreq_available(devfreq_path)
        original_max_freq = _read_devfreq_max(devfreq_path)
        print(f"[+] Jetson devfreq detected: {devfreq_path}")
        print(f"[+] Original max clock: {original_max_freq/1e6:.0f} MHz")
        print(f"[+] Available range: {available_freqs[0]/1e6:.0f}–{available_freqs[-1]/1e6:.0f} MHz")

        if clock_cap_mhz is not None:
            mit_freq = _snap_to_available(available_freqs, int(clock_cap_mhz * 1e6))
        else:
            mit_freq = _snap_to_available(available_freqs, int(original_max_freq * 0.4))
        print(f"[+] Mitigation clock cap: {mit_freq/1e6:.0f} MHz")

        mitigation_label = f"clock cap to {mit_freq/1e6:.0f} MHz"
        apply_mitigation = lambda: _set_devfreq_max(devfreq_path, mit_freq)
        restore_normal = lambda: _set_devfreq_max(devfreq_path, original_max_freq)
        restore_msg = f"sudo sh -c 'echo {original_max_freq} > {devfreq_path}/max_freq'"
    else:
        original_power_limit = get_current_power_limit(gpu_idx)
        print(f"[+] Original power limit: {original_power_limit}W")

        if mitigation_power_cap_w is None:
            mitigation_power_cap_w = original_power_limit * 0.7
        print(f"[+] Mitigation clamp target: {mitigation_power_cap_w:.0f}W")

        mitigation_label = f"power cap to {mitigation_power_cap_w:.0f}W"
        apply_mitigation = lambda: set_power_limit(mitigation_power_cap_w, gpu_idx)
        restore_normal = lambda: set_power_limit(original_power_limit, gpu_idx)
        restore_msg = f"sudo nvidia-smi -pl {int(original_power_limit)}"

    stress_cmd = [
        cuda_binary,
        "--mode", "oversubscribe",
        "--freq", "10.0",
        "--duty", "0.5",
        "--duration", str(duration_per_phase + 3.0)
    ]

    phases = [
        {"name": "attack_unmitigated",     "run_stress": True,  "mitigate": False},
        {"name": "attack_mitigated",       "run_stress": True,  "mitigate": True},
        {"name": "attack_terminated",      "run_stress": True,  "mitigate": False, "terminate": True},
        {"name": "victim_clamped_baseline", "run_stress": False, "mitigate": False, "force_clamp": True},
        {"name": "recovery_baseline",      "run_stress": False, "mitigate": False},
    ]

    all_trial_records = []
    per_trial_summaries = []

    try:
        for trial in range(1, num_trials + 1):
            detector = PlatformAgnosticAnomalyDetector(
                sampling_rate_hz=4.0,
                power_anomaly_threshold_w=detection_threshold_w
            )
            records, summary = _run_single_trial(
                trial_num=trial,
                total_trials=num_trials,
                assessor=assessor,
                detector=detector,
                cuda_binary=cuda_binary,
                stress_cmd=stress_cmd,
                phases=phases,
                duration_per_phase=duration_per_phase,
                apply_mitigation=apply_mitigation,
                restore_normal=restore_normal,
                mitigation_label=mitigation_label,
            )
            all_trial_records.extend(records)
            per_trial_summaries.append(summary)

    finally:
        print(f"\n[+] Restoring original settings")
        try:
            restore_normal()
        except Exception:
            print(f"    [!] WARNING: Could not restore. Run manually: {restore_msg}")

    if all_trial_records:
        fieldnames = list(all_trial_records[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trial_records)
        print(f"\n[+] T3 dataset saved to: {output_csv} ({len(all_trial_records)} samples across {num_trials} trials)")

    _print_aggregate_summary(per_trial_summaries, num_trials)

    return per_trial_summaries


def _print_aggregate_summary(per_trial_summaries, num_trials):
    """Prints aggregate statistics and mitigation decomposition across all trials."""
    phase_names = ["attack_unmitigated", "attack_mitigated", "attack_terminated", "victim_clamped_baseline", "recovery_baseline"]

    print("\n=====================================================================")
    print("      PROJECT GAIA: T3 CLOSED-LOOP MITIGATION SUMMARY (v2.0)        ")
    print(f"      Aggregated over {num_trials} trial(s)")
    print("=====================================================================")

    agg = {}
    for pn in phase_names:
        lats = [s[pn]["mean_total_latency_ms"] for s in per_trial_summaries if pn in s]
        pows = [s[pn]["mean_power_w"] for s in per_trial_summaries if pn in s]
        volts = [s[pn]["min_voltage_v"] for s in per_trial_summaries if pn in s]
        if lats:
            agg[pn] = {
                "lat_mean": np.mean(lats), "lat_std": np.std(lats, ddof=1) if len(lats) > 1 else 0.0,
                "pow_mean": np.mean(pows), "pow_std": np.std(pows, ddof=1) if len(pows) > 1 else 0.0,
                "volt_mean": np.mean(volts),
                "lat_values": lats,
            }

    print(f"\n  {'Phase':<28}{'Latency (ms)':<22}{'Power (W)':<18}{'Min V (V)':<12}")
    print("  " + "-" * 78)
    for pn in phase_names:
        if pn in agg:
            a = agg[pn]
            lat_str = f"{a['lat_mean']:.2f} +/- {a['lat_std']:.2f}" if num_trials > 1 else f"{a['lat_mean']:.2f}"
            pow_str = f"{a['pow_mean']:.2f} +/- {a['pow_std']:.2f}" if num_trials > 1 else f"{a['pow_mean']:.2f}"
            print(f"  {pn:<28}{lat_str:<22}{pow_str:<18}{a['volt_mean']:<12.4f}")

    print(f"\n  Power-Cap Enforcement Check:")
    if "attack_unmitigated" in agg and "attack_mitigated" in agg:
        p_unmit = agg["attack_unmitigated"]["pow_mean"]
        p_mit = agg["attack_mitigated"]["pow_mean"]
        print(f"    Unmitigated mean power: {p_unmit:.2f} W")
        print(f"    Mitigated mean power:   {p_mit:.2f} W")
        if p_mit < p_unmit:
            print(f"    Power reduction:        {p_unmit - p_mit:.2f} W ({(p_unmit - p_mit)/p_unmit*100:.1f}%)")
            print(f"    Cap enforcement:        CONFIRMED (power dropped under mitigation)")
        else:
            print(f"    Cap enforcement:        NOT CONFIRMED (mitigated power >= unmitigated)")

    if "attack_unmitigated" in agg and "attack_mitigated" in agg:
        A = agg["attack_unmitigated"]["lat_mean"]
        B = agg["attack_mitigated"]["lat_mean"]
        gross = A - B

        print(f"\n  Mitigation Decomposition:")
        print(f"    Unmitigated attack latency  (A) : {A:.2f} ms")
        print(f"    Mitigated attack latency    (B) : {B:.2f} ms")

        if "victim_clamped_baseline" in agg and "recovery_baseline" in agg:
            C = agg["victim_clamped_baseline"]["lat_mean"]
            D = agg["recovery_baseline"]["lat_mean"]
            overhead = C - D
            net = gross - overhead

            print(f"    Victim-alone clamped power  (C) : {C:.2f} ms")
            print(f"    Victim-alone full power     (D) : {D:.2f} ms")
            print(f"    Gross mitigation benefit  (A-B) : {gross:.2f} ms")
            print(f"    Clamping overhead          (C-D) : {overhead:.2f} ms")
            print(f"    Net attack mitigation (A-B)-(C-D): {net:.2f} ms")

        if num_trials >= 2:
            unmit_lats = agg["attack_unmitigated"]["lat_values"]
            mit_lats = agg["attack_mitigated"]["lat_values"]
            if len(unmit_lats) == len(mit_lats) and len(unmit_lats) >= 2:
                t_stat, p_val = ttest_rel(unmit_lats, mit_lats)
                sig = "YES" if p_val < 0.05 else "NO"
                print(f"\n  Paired t-test (unmitigated vs mitigated):")
                print(f"    t-statistic : {t_stat:.4f}")
                print(f"    p-value     : {p_val:.6f}")
                print(f"    Significant (alpha=0.05): {sig}")

    print("\n=====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project GAIA T3: Closed-Loop Power Mitigation")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration per phase in seconds")
    parser.add_argument("--cap", type=float, default=None,
                        help="Mitigation power cap in watts (desktop GPU, default: 70%% of current limit)")
    parser.add_argument("--clock-cap", type=float, default=None,
                        help="Mitigation clock cap in MHz (Jetson devfreq, default: 40%% of max)")
    parser.add_argument("--det-threshold", type=float, default=60.0,
                        help="Power anomaly detection threshold in watts (default: 60W)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--trials", type=int, default=5, help="Number of repeated trials (default: 5)")
    args = parser.parse_args()

    run_closed_loop_experiment(
        duration_per_phase=args.duration,
        mitigation_power_cap_w=args.cap,
        clock_cap_mhz=args.clock_cap,
        detection_threshold_w=args.det_threshold,
        output_csv=args.out,
        gpu_idx=args.gpu,
        num_trials=args.trials
    )
