import os
import sys
import time
import json
import random
import subprocess
import csv
import argparse
import platform
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gpu_fault_assessor_v2 import GPUFaultAssessorV2, CONDITION_LABELS, collect_platform_info

_EXE = ".exe" if sys.platform == "win32" else ""

JETSON_ARCH = {
    "thor": "sm_110",
    "orin": "sm_87",
    "xavier": "sm_72",
    "tx2": "sm_62",
    "nano": "sm_53",
}

def detect_gpu_arch():
    print("[arch] Detecting GPU compute capability...")

    # Quick check: Jetson device-tree (instant, no subprocess)
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().replace(b"\x00", b"").decode("utf-8", errors="replace").lower()
        for name, arch in JETSON_ARCH.items():
            if name in model:
                print(f"[arch] Jetson detected via device-tree: {arch} ({model.strip()})")
                return arch
    except (FileNotFoundError, PermissionError):
        pass

    # nvidia-smi (works on desktop and Thor; hangs on Orin/Xavier — timeout protects)
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
        out_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3" + _EXE)

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


def run_single_trial(phases, cuda_binary, duration_per_phase, assessor,
                     trial_label, randomize_order=False, victim_fn=None,
                     temp_sync_log=None, experiment_meta=None):
    """Run one complete trial through all phases.

    Args:
        victim_fn: callable returning dict with timing fields.
            Falls back to assessor.run_victim_workload_iteration().
        temp_sync_log: if a list, full thermal-sync metadata is appended
            per temp-sync phase (all readings, timestamps, trigger info).
        experiment_meta: dict of experiment-level metadata (victim_type,
            seed, etc.) included in each temp_sync_log entry.
    """
    records = []
    phase_summaries = {}
    trial_start = time.time()
    prev_current_a = 0.0
    prev_time = trial_start

    if victim_fn is None:
        victim_fn = assessor.run_victim_workload_iteration

    if randomize_order:
        phases = list(phases)
        random.shuffle(phases)

    for p_info in phases:
        p_name = p_info["name"]
        p_mode = p_info["mode"]
        p_freq = p_info["freq"]
        p_duty = p_info["duty"]
        temp_sync = p_info.get("temp_sync", False)
        temp_sync_trigger = "none"
        temp_sync_data = {}
        run_victim = p_info.get("run_victim", True)
        phase_victim_fn = p_info.get("victim_fn_override", victim_fn)
        stress_start_ts = None

        print(f"  [{trial_label}] Phase '{p_name}' (Mode: {p_mode}, Freq: {p_freq} kHz, Duty: {p_duty})...")
        cuda_proc = None

        label_info = CONDITION_LABELS.get(p_name)
        if label_info is None:
            is_atk = p_mode is not None
            label_info = {"is_attack": is_atk,
                          "category": "sweep" if p_name.startswith("sweep_") else "unknown"}
        is_attack = int(label_info["is_attack"])

        if temp_sync and p_mode is not None:
            from gpu_fault_assessor_v2 import _cuda_subprocess_env

            preheat_start = time.time()
            print(f"    [temp-sync] Phase 1: Pre-heating GPU (5s stress burst)...")
            preheat_cmd = [
                cuda_binary, "--mode", "oversubscribe",
                "--freq", "0.001", "--duty", "1.0", "--duration", "6"
            ]
            preheat_proc = subprocess.Popen(
                preheat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=_cuda_subprocess_env()
            )
            time.sleep(5)
            preheat_proc.terminate()
            try:
                preheat_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                preheat_proc.kill()
            heated_temp = assessor.get_hardware_telemetry().get("gpu_temp_c", 0)
            preheat_end = time.time()
            print(f"    [temp-sync] Post-heat temp: {heated_temp:.3f}C")

            print(f"    [temp-sync] Phase 2: Cooling (3s idle)...")
            cooldown_start = time.time()
            time.sleep(3)
            cooldown_end = time.time()
            cooled_temp = assessor.get_hardware_telemetry().get("gpu_temp_c", 0)
            print(f"    [temp-sync] Post-cool temp: {cooled_temp:.3f}C (dropped {heated_temp - cooled_temp:.3f}C)")

            print(f"    [temp-sync] Phase 3: Watching for thermal inflection...")
            prev_temp = cooled_temp
            temp_trace = [(time.time() - trial_start, cooled_temp)]
            sync_start = time.time()

            victim_started = run_victim
            if run_victim:
                phase_victim_fn()

            while (time.time() - sync_start) < 15.0:
                if run_victim:
                    phase_victim_fn()
                else:
                    time.sleep(0.25)
                cur_temp = assessor.get_hardware_telemetry().get("gpu_temp_c", 0)
                temp_trace.append((time.time() - trial_start, cur_temp))
                if len(temp_trace) >= 3:
                    t2 = temp_trace[-1][1]
                    t1 = temp_trace[-2][1]
                    t0 = temp_trace[-3][1]
                    if t2 >= t1 and t1 <= t0 and (t2 - t1) > 0:
                        elapsed_sync = time.time() - sync_start
                        print(f"    [temp-sync] Inflection: {t1:.3f}C -> {t2:.3f}C "
                              f"after {elapsed_sync:.1f}s")
                        temp_sync_trigger = "temp_triggered"
                        break
                prev_temp = cur_temp
                time.sleep(0.05)
            else:
                cur_temp = assessor.get_hardware_telemetry().get("gpu_temp_c", 0)
                print(f"    [temp-sync] Timeout after 15s (temp: {cur_temp:.3f}C)")
                temp_sync_trigger = "timeout_triggered"

            temp_sync_data = {
                "trial": trial_label,
                "phase": p_name,
                "preheat_start_s": round(preheat_start - trial_start, 3),
                "preheat_end_s": round(preheat_end - trial_start, 3),
                "heated_temp_c": heated_temp,
                "cooldown_start_s": round(cooldown_start - trial_start, 3),
                "cooldown_end_s": round(cooldown_end - trial_start, 3),
                "cooled_temp_c": cooled_temp,
                "watch_start_s": round(sync_start - trial_start, 3),
                "trigger": temp_sync_trigger,
                "detection_delay_s": round(time.time() - sync_start, 3),
                "run_victim": run_victim,
                "temp_trace": [(round(t, 4), round(v, 4)) for t, v in temp_trace],
            }
            if experiment_meta:
                temp_sync_data["experiment_meta"] = experiment_meta
            if temp_sync_log is not None:
                temp_sync_log.append(temp_sync_data)

        if p_mode is not None and os.path.exists(cuda_binary):
            cmd = [
                cuda_binary,
                "--mode", p_mode,
                "--freq", str(p_freq),
                "--duty", str(p_duty),
                "--duration", str(duration_per_phase + 3.0)
            ]
            try:
                from gpu_fault_assessor_v2 import _cuda_subprocess_env
                cuda_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_cuda_subprocess_env())
                stress_start_ts = time.time()
                time.sleep(0.5)
            except Exception as e:
                raise RuntimeError(f"Failed to spawn stress process: {e}")

        latencies_tot = []
        latencies_delay = []
        latencies_compute = []
        phase_start = time.time()

        while (time.time() - phase_start) < duration_per_phase:
            if run_victim:
                v_res = phase_victim_fn()
            else:
                v_res = {"total_latency_ms": 0, "launch_delay_ms": 0,
                         "gpu_compute_ms": 0, "victim_backend": "none"}
                time.sleep(0.1)
            latencies_tot.append(v_res["total_latency_ms"])
            latencies_delay.append(v_res["launch_delay_ms"])
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

            sm_clk = hw.get("sm_clock_mhz", 0)
            gpu_temp = hw.get("gpu_temp_c", 0)

            records.append({
                "timestamp": round(t_curr - trial_start, 4),
                "phase": p_name,
                "is_attack": is_attack,
                "category": label_info["category"],
                "data_source": hw["provenance"],
                "power_w": round(p_w, 3),
                "voltage_v": round(v_meas, 5),
                "current_a": round(curr_a, 4),
                "di_dt_amps_per_sec": round(di_dt, 4),
                "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
                "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                "victim_launch_delay_ms": round(v_res["launch_delay_ms"], 3),
                "victim_backend": v_res.get("victim_backend", "unknown"),
                "sm_clock_mhz": sm_clk,
                "gpu_temp_c": gpu_temp,
                "gpu_util_pct": hw.get("gpu_util_pct"),
                "mem_util_pct": hw.get("mem_util_pct"),
                "sw_power_cap": hw.get("sw_power_cap", 0),
                "hw_slowdown": hw.get("hw_slowdown", 0),
                "throttle_bitmask": hw.get("throttle_bitmask", 0),
                "temp_sync_trigger": temp_sync_trigger,
                "stress_start_ts": round(stress_start_ts - trial_start, 4) if stress_start_ts else None,
            })

            prev_current_a = curr_a
            prev_time = t_curr
            time.sleep(0.02)

        stress_stop_ts = None
        if cuda_proc is not None:
            cuda_proc.terminate()
            stress_stop_ts = time.time()
            try:
                cuda_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                cuda_proc.kill()
        for r in records:
            if r["phase"] == p_name and stress_stop_ts:
                r["stress_stop_ts"] = round(stress_stop_ts - trial_start, 4)

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
    randomize_order=False,
    arch_override=None,
    seed=None,
    victim_type="synthetic",
):
    if cuda_binary is None:
        cuda_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3" + _EXE)
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "controlled_3phase_experiment_v2.csv")

    summary_csv = output_csv.replace(".csv", "_summary.csv")
    output_dir = os.path.dirname(output_csv) or "."

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    print("=====================================================================")
    print(" Project GAIA: Controlled Multi-Phase Experiment Harness (v2.1)       ")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s | Seed: {seed}")
    print(f" Victim: {victim_type}")
    print("=====================================================================")

    if not os.path.exists(cuda_binary):
        compile_cuda_backend(out_binary=cuda_binary, arch_override=arch_override)

    print("\n[init] Initializing hardware assessor...")
    assessor = GPUFaultAssessorV2(arch_override=arch_override)
    print("[init] Assessor ready")

    platform_info = collect_platform_info()
    platform_info["experiment_seed"] = seed
    platform_info["victim_type"] = victim_type
    platform_info["num_trials"] = num_trials
    platform_info["duration_per_phase"] = duration_per_phase
    platform_info["randomize_order"] = randomize_order
    platform_info["sweep_mechanism"] = sweep_mechanism
    platform_path = os.path.join(output_dir, "platform_info.json")
    with open(platform_path, "w") as f:
        json.dump(platform_info, f, indent=2, default=str)
    print(f"[init] Platform info saved to {platform_path}")

    from realistic_victims import create_victim, create_legitimate_workload
    victim_fn = create_victim(victim_type)
    legit_workload_fn = create_legitimate_workload()
    print(f"[init] Victim ({victim_type}) and legitimate workload ready\n")

    phases = [
        {"name": "victim_alone", "mode": None, "freq": 0, "duty": 0.0},
        {"name": "victim_sustained_stress", "mode": "oversubscribe", "freq": 0.001, "duty": 1.0},
        {"name": "victim_cache_only", "mode": "cache", "freq": 0.001, "duty": 1.0},
        {"name": "victim_compute_only", "mode": "compute", "freq": 0.001, "duty": 1.0},
        {"name": "victim_pulsed_stress", "mode": "oversubscribe", "freq": 10.0, "duty": 0.5},
        {"name": "victim_temp_sync_stress", "mode": "oversubscribe", "freq": 10.0, "duty": 0.5,
         "temp_sync": True},
        {"name": "victim_occupancy_stress", "mode": "occupancy", "freq": 0.001, "duty": 1.0},
        {"name": "victim_legit_highpower", "mode": None, "freq": 0, "duty": 0.0,
         "victim_fn_override": legit_workload_fn},
        {"name": "temp_sync_control_novictim", "mode": "oversubscribe", "freq": 10.0, "duty": 0.5,
         "temp_sync": True, "run_victim": False},
        {"name": "temp_sync_randomized_delay", "mode": "oversubscribe", "freq": 10.0, "duty": 0.5,
         "temp_sync": True},
    ]

    if sweep_mechanism:
        # Frequency sweep at fixed 50% duty: tests DVFS interaction
        # from grid-relevant sub-Hz up through kHz
        sweep_freq_hz = [0.5, 1, 2, 5, 10, 50, 100, 500, 10000]
        for f_hz in sweep_freq_hz:
            freq_khz = f_hz / 1000.0
            phases.append({
                "name": f"sweep_freq_{f_hz}hz",
                "mode": "oversubscribe",
                "freq": freq_khz,
                "duty": 0.5
            })
        # Duty-cycle sweep at fixed 10 kHz: tests power vs active fraction
        for duty_pct in [25, 50, 75]:
            phases.append({
                "name": f"sweep_duty_{duty_pct}pct",
                "mode": "oversubscribe",
                "freq": 10.0,
                "duty": duty_pct / 100.0,
            })

    print("\n[+] Warmup trial (discarded)...")
    run_single_trial(phases, cuda_binary, duration_per_phase, assessor,
                     "warmup", victim_fn=victim_fn)
    print("[+] Warmup complete. Starting recorded trials.\n")

    all_records = []
    all_trial_summaries = []
    per_phase_trial_means = {}
    temp_sync_log = []

    experiment_meta = {
        "victim_type": victim_type,
        "seed": seed,
        "num_trials": num_trials,
        "duration_per_phase": duration_per_phase,
    }

    for phase in phases:
        per_phase_trial_means[phase["name"]] = []

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'='*60}")
        print(f"  Trial {trial_idx}/{num_trials}")
        print(f"{'='*60}")

        records, phase_summaries = run_single_trial(
            phases, cuda_binary, duration_per_phase, assessor, f"T{trial_idx}",
            randomize_order=randomize_order,
            victim_fn=victim_fn,
            temp_sync_log=temp_sync_log,
            experiment_meta=experiment_meta,
        )

        for r in records:
            r["trial"] = trial_idx
        all_records.extend(records)

        for p_name, stats in phase_summaries.items():
            per_phase_trial_means[p_name].append(stats["mean_total_latency_ms"])
            row = {"trial": trial_idx, "phase": p_name}
            row.update(stats)
            all_trial_summaries.append(row)

    if temp_sync_log:
        csv_stem = os.path.splitext(os.path.basename(output_csv))[0]
        ts_path = os.path.join(output_dir, f"{csv_stem}_temp_sync.json")
        with open(ts_path, "w") as f:
            json.dump(temp_sync_log, f, indent=2)
        print(f"[+] Temp-sync trace log saved to: {ts_path}")

    if all_records:
        fieldnames = [
            "trial", "timestamp", "phase", "is_attack", "category",
            "data_source", "power_w", "voltage_v",
            "current_a", "di_dt_amps_per_sec", "victim_kernel_ms",
            "victim_total_latency_ms", "victim_launch_delay_ms",
            "victim_backend",
            "sm_clock_mhz", "gpu_temp_c", "gpu_util_pct", "mem_util_pct",
            "sw_power_cap", "hw_slowdown", "throttle_bitmask",
            "temp_sync_trigger", "stress_start_ts", "stress_stop_ts"
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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

    core_phases = ["victim_alone", "victim_sustained_stress", "victim_pulsed_stress"]
    has_all_core = all(p in per_phase_trial_means and len(per_phase_trial_means[p]) == num_trials for p in core_phases)

    if has_all_core and num_trials >= 2:
        alone = np.array(per_phase_trial_means["victim_alone"])
        sustained = np.array(per_phase_trial_means["victim_sustained_stress"])
        pulsed = np.array(per_phase_trial_means["victim_pulsed_stress"])

        t_ps, p_ps = ttest_rel(pulsed, sustained)
        t_pa, p_pa = ttest_rel(pulsed, alone)
        t_sa, p_sa = ttest_rel(sustained, alone)

        print(f"  pulsed_stress vs sustained_stress: t={t_ps:>7.3f}, p={p_ps:.6f} {significance_stars(p_ps)}")
        print(f"  pulsed_stress vs victim_alone    : t={t_pa:>7.3f}, p={p_pa:.6f} {significance_stars(p_pa)}")
        print(f"  sustained_stress vs victim_alone : t={t_sa:>7.3f}, p={p_sa:.6f} {significance_stars(p_sa)}")

        if "victim_cache_only" in per_phase_trial_means and "victim_compute_only" in per_phase_trial_means:
            cache = np.array(per_phase_trial_means["victim_cache_only"])
            compute = np.array(per_phase_trial_means["victim_compute_only"])
            t_cc, p_cc = ttest_rel(cache, compute)
            print(f"  cache_only vs compute_only       : t={t_cc:>7.3f}, p={p_cc:.6f} {significance_stars(p_cc)}")
    elif num_trials < 2:
        print("  (skipped — need >= 2 trials for paired t-test)")
    else:
        print("  (skipped — not all core phases present)")

    print("\n Power-Match Verification:")
    for pn in ["victim_sustained_stress", "victim_pulsed_stress"]:
        if pn in aggregate:
            print(f"  {pn}: {aggregate[pn]['power']:.2f} W mean")
    if "victim_sustained_stress" in aggregate and "victim_pulsed_stress" in aggregate:
        ratio = aggregate["victim_sustained_stress"]["power"] / max(aggregate["victim_pulsed_stress"]["power"], 0.01)
        matched = "YES" if 0.8 <= ratio <= 1.2 else "NO"
        print(f"  Ratio (sustained/pulsed): {ratio:.3f}  Power-matched: {matched}")

    print("=====================================================================\n")

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project GAIA Thrust 1 Experiment Runner")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration per phase in seconds")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: alongside this script)")
    parser.add_argument("--sweep", action="store_true", help="Include frequency/duty-cycle sweep phases")
    parser.add_argument("--trials", type=int, default=10, help="Number of recorded trials (default: 10)")
    parser.add_argument("--arch", type=str, default=None, help="GPU arch override, e.g. sm_110 for Jetson Thor")
    parser.add_argument("--randomize", action="store_true", help="Randomize phase order per trial")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--victim-type", type=str, default="synthetic",
                        choices=["synthetic", "training", "inference"],
                        help="Victim workload type (default: synthetic)")
    args = parser.parse_args()

    run_controlled_3phase_experiment(
        duration_per_phase=args.duration,
        output_csv=args.out,
        sweep_mechanism=args.sweep,
        num_trials=args.trials,
        arch_override=args.arch,
        randomize_order=args.randomize,
        seed=args.seed,
        victim_type=args.victim_type,
    )
