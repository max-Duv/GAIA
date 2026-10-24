import os
import sys
import time
import subprocess
import csv
import argparse
import platform
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gpu_fault_assessor_v2 import GPUFaultAssessorV2

JETSON_ARCH = {
    "orin": "sm_87",
    "xavier": "sm_72",
    "tx2": "sm_62",
    "nano": "sm_53",
}

def detect_gpu_arch():
    print("[arch] Detecting GPU compute capability...")

    # Jetson kernels contain "tegra"; nvidia-smi and pip torch both hang there.
    is_tegra = "tegra" in platform.release().lower()

    if is_tegra:
        # Device-tree model string has a trailing null byte — read as binary
        try:
            with open("/proc/device-tree/model", "rb") as f:
                model = f.read().replace(b"\x00", b"").decode("utf-8", errors="replace").lower()
            for name, arch in JETSON_ARCH.items():
                if name in model:
                    print(f"[arch] Jetson detected: {arch} ({model.strip()})")
                    return arch
            print(f"[arch] Jetson device-tree found but unrecognized model: {model.strip()}")
        except (FileNotFoundError, PermissionError) as e:
            print(f"[arch] Tegra kernel but no device-tree: {e}")
        raise RuntimeError(
            "Jetson detected but model unknown. Use --arch sm_XX (e.g. --arch sm_87 for Orin)."
        )

    # Desktop path: nvidia-smi
    print("[arch] Trying nvidia-smi...")
    try:
        proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            stdout, _ = proc.communicate(timeout=5)
            if proc.returncode == 0:
                cap = stdout.strip().split("\n")[0].strip()
                arch = f"sm_{cap.replace('.', '')}"
                print(f"[arch] Detected via nvidia-smi: {arch}")
                return arch
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            print("[arch] nvidia-smi timed out")
    except FileNotFoundError:
        print("[arch] nvidia-smi not found")

    raise RuntimeError(
        "Cannot detect GPU compute capability. Use --arch sm_XX to specify manually."
    )


def compile_cuda_backend(src_path=None, out_binary=None, arch_override=None):
    if src_path is None:
        src_path = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3.cu")
    if out_binary is None:
        out_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"CUDA source file not found: {src_path}")

    if arch_override:
        arch = arch_override
        print(f"[arch] Using --arch override: {arch}")
    else:
        arch = detect_gpu_arch()
    print(f"GPU architecture: {arch}")

    nvcc_cmd = ["nvcc", "-O3", f"-arch={arch}", src_path, "-o", out_binary, "-lcuda"]
    try:
        res = subprocess.run(nvcc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(f"Compiled CUDA stress backend: '{out_binary}'")
            return True
        else:
            raise RuntimeError(f"nvcc compilation failed:\n{res.stderr}")
    except FileNotFoundError:
        raise RuntimeError("nvcc not found. Check to ensure CUDA toolkit is installed and nvcc is on PATH.")


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"


def run_single_trial(phases, cuda_binary, duration_per_phase, assessor, trial_label):
    V_nominal = 1.0
    R_parasitic = 0.0018
    L_parasitic = 1.25e-7

    records = []
    phase_summaries = {}
    trial_start = time.time()
    prev_power = 0.0
    prev_time = trial_start

    for p_info in phases:
        p_name = p_info["name"]
        p_mode = p_info["mode"]
        p_freq = p_info["freq"]
        p_duty = p_info["duty"]

        print(f"  [{trial_label}] Phase '{p_name}' (Mode: {p_mode}, Freq: {p_freq} kHz, Duty: {p_duty})...")
        cuda_proc = None

        if p_mode is not None and os.path.exists(cuda_binary):
            cmd = [
                cuda_binary,
                "--mode", p_mode,
                "--freq", str(p_freq),
                "--duty", str(p_duty),
                "--duration", str(duration_per_phase + 3.0)
            ]
            try:
                cuda_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                time.sleep(0.5)
            except Exception as e:
                raise RuntimeError(f"Failed to spawn stress process: {e}")

        latencies_tot = []
        latencies_delay = []
        latencies_compute = []
        phase_start = time.time()

        while (time.time() - phase_start) < duration_per_phase:
            v_res = assessor.run_victim_workload_iteration()
            latencies_tot.append(v_res["total_latency_ms"])
            latencies_delay.append(v_res["launch_delay_ms"])
            latencies_compute.append(v_res["gpu_compute_ms"])

            t_curr = time.time()
            dt = max(t_curr - prev_time, 0.001)
            hw = assessor.get_hardware_telemetry()

            p_w = hw["power_w"]
            v_meas = hw.get("voltage_v", V_nominal)

            curr_a = p_w / max(v_meas, 0.5)
            prev_curr_a = prev_power / V_nominal
            di_dt = (curr_a - prev_curr_a) / dt
            v_drop_ir = curr_a * R_parasitic
            v_drop_l = L_parasitic * di_dt
            v_modeled = V_nominal - v_drop_ir - v_drop_l

            records.append({
                "timestamp": round(t_curr - trial_start, 4),
                "phase": p_name,
                "data_source": hw["provenance"],
                "power_w": round(p_w, 3),
                "voltage_v": round(v_meas if hw["provenance"] == "Jetson_INA3221_Hardware" else v_modeled, 5),
                "current_a": round(curr_a, 3),
                "di_dt_amps_per_sec": round(di_dt, 2),
                "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                "victim_launch_delay_ms": round(v_res["launch_delay_ms"], 3),
                "victim_gpu_compute_ms": round(v_res["gpu_compute_ms"], 3),
                "sw_power_cap": hw.get("sw_power_cap", 0),
                "hw_slowdown": hw.get("hw_slowdown", 0),
                "throttle_bitmask": hw.get("throttle_bitmask", 0)
            })

            prev_power = p_w
            prev_time = t_curr
            time.sleep(0.02)

        if cuda_proc is not None:
            cuda_proc.terminate()
            try:
                cuda_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cuda_proc.kill()

        phase_records = [r for r in records if r["phase"] == p_name]
        cap_count = sum(1 for r in phase_records if r["sw_power_cap"] == 1 or r["hw_slowdown"] == 1)

        phase_summaries[p_name] = {
            "mean_total_latency_ms": float(np.mean(latencies_tot)),
            "std_total_latency_ms": float(np.std(latencies_tot, ddof=1)) if len(latencies_tot) > 1 else 0.0,
            "p50_total_latency_ms": float(np.percentile(latencies_tot, 50)),
            "p95_total_latency_ms": float(np.percentile(latencies_tot, 95)),
            "p99_total_latency_ms": float(np.percentile(latencies_tot, 99)),
            "mean_launch_delay_ms": float(np.mean(latencies_delay)),
            "std_launch_delay_ms": float(np.std(latencies_delay, ddof=1)) if len(latencies_delay) > 1 else 0.0,
            "mean_gpu_compute_ms": float(np.mean(latencies_compute)),
            "std_gpu_compute_ms": float(np.std(latencies_compute, ddof=1)) if len(latencies_compute) > 1 else 0.0,
            "mean_power_w": float(np.mean([r["power_w"] for r in phase_records])),
            "std_power_w": float(np.std([r["power_w"] for r in phase_records], ddof=1)) if len(phase_records) > 1 else 0.0,
            "min_voltage_v": float(np.min([r["voltage_v"] for r in phase_records])),
            "capping_rate_pct": (cap_count / max(len(phase_records), 1)) * 100.0,
            "n_samples": len(phase_records)
        }

        s = phase_summaries[p_name]
        print(f"    Latency: {s['mean_total_latency_ms']:.2f} ± {s['std_total_latency_ms']:.2f} ms | "
              f"Power: {s['mean_power_w']:.2f} W | Cap: {s['capping_rate_pct']:.1f}%")

    return records, phase_summaries


def run_controlled_3phase_experiment(
    cuda_binary=None,
    duration_per_phase=15.0,
    output_csv=None,
    sweep_mechanism=False,
    num_trials=10,
    arch_override=None
):
    """
    Executes a controlled 3-condition experiment with repeated trials for statistical rigor.

    Conditions:
    1. 'victim_alone' — Baseline un-contended GPU
    2. 'victim_matched_benign' — Victim + continuous compute stress
    3. 'victim_pulsed_stress' — Victim + 10.0 kHz frequency-pulsed stress

    Runs a warmup trial (discarded), then num_trials recorded trials.
    Produces per-sample CSV, per-trial summary CSV, and paired t-tests.
    """
    if cuda_binary is None:
        cuda_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3")
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "controlled_3phase_experiment_v2.csv")

    summary_csv = output_csv.replace(".csv", "_summary.csv")

    print("=====================================================================")
    print(" Project GAIA: Controlled 3-Phase & Mechanism-Isolation Harness (v2.0) ")
    print(" Grounded in RTSS (MPE), Islam/Ren, & Jetson Orin INA3221 Telemetry   ")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s           ")
    print("=====================================================================")

    if not os.path.exists(cuda_binary):
        compile_cuda_backend(out_binary=cuda_binary, arch_override=arch_override)

    print("\n[init] Initializing hardware assessor...")
    assessor = GPUFaultAssessorV2()
    print("[init] Assessor ready\n")

    phases = [
        {"name": "victim_alone", "mode": None, "freq": 0, "duty": 0.0},
        {"name": "victim_matched_benign", "mode": "compute", "freq": 0.001, "duty": 1.0},
        {"name": "victim_pulsed_stress", "mode": "oversubscribe", "freq": 10.0, "duty": 0.5}
    ]

    if sweep_mechanism:
        sweep_windows_ms = [1, 5, 10, 50, 100, 250, 500]
        for w_ms in sweep_windows_ms:
            phases.append({
                "name": f"sweep_window_{w_ms}ms",
                "mode": "oversubscribe",
                "freq": 1000.0 / (2.0 * w_ms),
                "duty": 0.5
            })

    # --- Warmup trial (discarded) to stabilize GPU clocks and thermals ---
    print("\n[+] Warmup trial (discarded)...")
    run_single_trial(phases, cuda_binary, duration_per_phase, assessor, "warmup")
    print("[+] Warmup complete. Starting recorded trials.\n")

    # --- Recorded trials ---
    all_records = []
    all_trial_summaries = []  # list of {trial, phase, ...stats}
    per_phase_trial_means = {}  # phase_name -> list of per-trial mean latencies

    for phase in phases:
        per_phase_trial_means[phase["name"]] = []

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'='*60}")
        print(f"  Trial {trial_idx}/{num_trials}")
        print(f"{'='*60}")

        records, phase_summaries = run_single_trial(
            phases, cuda_binary, duration_per_phase, assessor, f"T{trial_idx}"
        )

        for r in records:
            r["trial"] = trial_idx
        all_records.extend(records)

        for p_name, stats in phase_summaries.items():
            per_phase_trial_means[p_name].append(stats["mean_total_latency_ms"])
            row = {"trial": trial_idx, "phase": p_name}
            row.update(stats)
            all_trial_summaries.append(row)

    if all_records:
        fieldnames = [
            "trial", "timestamp", "phase", "data_source", "power_w", "voltage_v",
            "current_a", "di_dt_amps_per_sec", "victim_total_latency_ms",
            "victim_launch_delay_ms", "victim_gpu_compute_ms", "sw_power_cap",
            "hw_slowdown", "throttle_bitmask"
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[+] Per-sample dataset saved to: {output_csv}")

    if all_trial_summaries:
        sum_fieldnames = list(all_trial_summaries[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sum_fieldnames)
            writer.writeheader()
            writer.writerows(all_trial_summaries)
        print(f"[+] Per-trial summary saved to: {summary_csv}")

    print("\n=====================================================================")
    print("      PROJECT GAIA: THRUST 1 CONTROLLED TRIAL SUMMARY (v2.0)        ")
    print(f"      {num_trials} trials x {duration_per_phase}s per phase              ")
    print("=====================================================================")
    print(f"{'Phase':<26}{'Latency(ms)':<18}{'Std':<10}{'p50':<10}{'p95':<10}{'p99':<10}{'Power(W)':<12}{'Cap%':<8}")
    print("-" * 104)

    aggregate = {}
    for p_info in phases:
        p_name = p_info["name"]
        trial_rows = [s for s in all_trial_summaries if s["phase"] == p_name]

        lat_means = [s["mean_total_latency_ms"] for s in trial_rows]
        lat_stds = [s["std_total_latency_ms"] for s in trial_rows]
        pow_means = [s["mean_power_w"] for s in trial_rows]
        cap_rates = [s["capping_rate_pct"] for s in trial_rows]
        min_volts = [s["min_voltage_v"] for s in trial_rows]
        p50s = [s["p50_total_latency_ms"] for s in trial_rows]
        p95s = [s["p95_total_latency_ms"] for s in trial_rows]
        p99s = [s["p99_total_latency_ms"] for s in trial_rows]

        agg_mean = float(np.mean(lat_means))
        agg_std = float(np.std(lat_means, ddof=1)) if len(lat_means) > 1 else 0.0
        agg_p50 = float(np.mean(p50s))
        agg_p95 = float(np.mean(p95s))
        agg_p99 = float(np.mean(p99s))
        agg_pow = float(np.mean(pow_means))
        agg_cap = float(np.mean(cap_rates))
        agg_min_v = float(np.min(min_volts))

        aggregate[p_name] = {
            "mean": agg_mean, "std": agg_std,
            "p50": agg_p50, "p95": agg_p95, "p99": agg_p99,
            "power": agg_pow, "cap": agg_cap, "min_v": agg_min_v
        }

        print(f"{p_name:<26}{agg_mean:>6.2f} ± {agg_std:<6.2f}  {agg_std:<10.2f}{agg_p50:<10.2f}{agg_p95:<10.2f}{agg_p99:<10.2f}{agg_pow:<12.2f}{agg_cap:<8.1f}")

    # --- Paired t-tests ---
    print("\n Paired t-test across trial means)")

    core_phases = ["victim_alone", "victim_matched_benign", "victim_pulsed_stress"]
    has_all_core = all(p in per_phase_trial_means and len(per_phase_trial_means[p]) == num_trials for p in core_phases)

    if has_all_core and num_trials >= 2:
        alone = np.array(per_phase_trial_means["victim_alone"])
        benign = np.array(per_phase_trial_means["victim_matched_benign"])
        pulsed = np.array(per_phase_trial_means["victim_pulsed_stress"])

        t_pb, p_pb = ttest_rel(pulsed, benign)
        t_pa, p_pa = ttest_rel(pulsed, alone)
        t_ba, p_ba = ttest_rel(benign, alone)

        print(f"  pulsed_stress vs matched_benign : t={t_pb:>7.3f}, p={p_pb:.6f} {significance_stars(p_pb)}")
        print(f"  pulsed_stress vs victim_alone   : t={t_pa:>7.3f}, p={p_pa:.6f} {significance_stars(p_pa)}")
        print(f"  matched_benign vs victim_alone  : t={t_ba:>7.3f}, p={p_ba:.6f} {significance_stars(p_ba)}")
    elif num_trials < 2:
        print("  (skipped — need >= 2 trials for paired t-test)")
    else:
        print("  (skipped — not all core phases present)")

    print("=====================================================================\n")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project GAIA Thrust 1 Experiment Runner")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration per phase in seconds")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: alongside this script)")
    parser.add_argument("--sweep", action="store_true", help="Include launch-duration sweep for DAC Fig 6")
    parser.add_argument("--trials", type=int, default=10, help="Number of recorded trials (default: 10)")
    parser.add_argument("--arch", type=str, default=None, help="GPU arch override, e.g. sm_87 for Jetson Orin (skips detection)")
    args = parser.parse_args()

    run_controlled_3phase_experiment(
        duration_per_phase=args.duration,
        output_csv=args.out,
        sweep_mechanism=args.sweep,
        num_trials=args.trials,
        arch_override=args.arch
    )
