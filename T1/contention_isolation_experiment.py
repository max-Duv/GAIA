"""
Project GAIA: Contention-vs-Power Isolation Experiment
======================================================
Addresses cross-examiner finding G03: victim slowdown could be explained
by memory/scheduler contention alone without any power-domain effect.

Experimental design — three matched-load comparison pairs:

Pair 1 (same resource profile, different power):
  - compute-only attack (moderate power ~62W, high SM occupancy)
  - oversubscribe attack (high power ~86W, similar SM occupancy)
  If victim degrades MORE under oversubscribe after controlling for
  occupancy, that implicates a power-domain mechanism.

Pair 2 (same power level, different resource profile):
  - cache-only at 100% duty (high mem bandwidth, high power)
  - compute-only at tuned duty to match cache power level
  If degradation differs despite matched power, that implicates
  resource contention as the dominant mechanism.

Pair 3 (DVFS isolation):
  - Each attack condition run under both DEFAULT and FIXED clock
  - Separates DVFS-induced effects from direct contention
  - Fixed clock removes the governor's frequency response to load

Each condition includes a clock-matched victim-only baseline measured
at the same devfreq state, so degradation is always computed against
the correct reference.

Usage (on Jetson):
    sudo python contention_isolation_experiment.py [--trials 10] [--duration 15]
"""

import os
import sys
import time
import json
import random
import subprocess
import csv
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gpu_fault_assessor_v2 import GPUFaultAssessorV2, CONDITION_LABELS, collect_platform_info, _cuda_subprocess_env
from controlled_3phase_runner_v2 import (
    compile_cuda_backend, detect_gpu_arch,
    devfreq_snapshot, devfreq_set_fixed, devfreq_restore,
    devfreq_pick_fixed_freq, _read_devfreq,
)

_EXE = ".exe" if sys.platform == "win32" else ""


# Each condition: (name, cuda_mode, freq_khz, duty_cycle, description)
# Pair 1: same resource type, different power
# Pair 2: same power target, different resource type
# Pair 3 is the DVFS dimension — each condition above runs under both clock policies
CONDITIONS = [
    # -- Baselines --
    ("baseline_idle",       None,           0,      0.0,
     "No attacker, victim only"),

    # -- Pair 1: similar occupancy, different power --
    ("pair1_compute",       "compute",      0.001,  1.0,
     "Compute-only sustained (moderate power, high SM)"),
    ("pair1_oversubscribe", "oversubscribe", 0.001, 1.0,
     "Cache+compute interleaved (high power, similar SM)"),

    # -- Pair 2: matched power, different resource profile --
    ("pair2_cache_full",    "cache",        0.001,  1.0,
     "Cache-only sustained (high mem BW, high power)"),
    ("pair2_compute_tuned", "compute",      0.001,  0.75,
     "Compute at 75% duty — tuned to approximate cache power level"),

    # -- Additional controls --
    ("ctrl_occupancy",      "occupancy",    0.001,  1.0,
     "Occupancy saturation (shared-mem pressure, moderate power)"),
    ("ctrl_pulsed",         "oversubscribe", 10.0,  0.5,
     "Pulsed oversubscribe at 10kHz 50% — temporal modulation"),
]


def _launch_stress(cuda_binary, mode, freq_khz, duty, duration_s):
    """Launch the CUDA stress binary as a subprocess. Returns (proc, pid)."""
    cmd = [
        cuda_binary,
        "--mode", mode,
        "--freq", str(freq_khz),
        "--duty", str(duty),
        "--duration", str(duration_s + 3.0),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=_cuda_subprocess_env()
    )
    time.sleep(0.5)
    return proc


def _collect_phase(assessor, victim_fn, duration_s, trial_start, phase_name,
                   is_attack, clock_policy, cuda_proc_pid=None):
    """Run victim iterations for duration_s, collecting telemetry each iteration.
    Returns list of record dicts."""
    records = []
    prev_current_a = 0.0
    prev_time = time.time()
    phase_start = time.time()

    while (time.time() - phase_start) < duration_s:
        v_res = victim_fn()

        t_curr = time.time()
        dt = max(t_curr - prev_time, 0.001)
        hw = assessor.get_hardware_telemetry()

        p_w = hw["power_w"]
        v_meas = hw.get("voltage_v", 0.0)
        if v_meas <= 0:
            v_meas = 19.75
        curr_a = p_w / v_meas
        di_dt = (curr_a - prev_current_a) / dt

        devfreq_cur = hw.get("devfreq_cur_freq_hz")
        if devfreq_cur is None and assessor._devfreq_path:
            try:
                devfreq_cur = int(_read_devfreq(assessor._devfreq_path, 'cur_freq'))
            except (IOError, OSError, ValueError):
                devfreq_cur = None

        records.append({
            "timestamp": round(t_curr - trial_start, 4),
            "phase": phase_name,
            "is_attack": int(is_attack),
            "clock_policy": clock_policy,
            "power_w": round(p_w, 3),
            "voltage_v": round(v_meas, 5),
            "current_a": round(curr_a, 4),
            "di_dt_amps_per_sec": round(di_dt, 4),
            "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
            "victim_gpu_compute_ms": round(v_res["gpu_compute_ms"], 3),
            "victim_launch_delay_ms": round(v_res["launch_delay_ms"], 3),
            "victim_backend": v_res.get("victim_backend", "unknown"),
            "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
            "gpu_temp_c": hw.get("gpu_temp_c", 0),
            "gpu_util_pct": hw.get("gpu_util_pct"),
            "mem_util_pct": hw.get("mem_util_pct"),
            "sw_power_cap": hw.get("sw_power_cap", 0),
            "hw_slowdown": hw.get("hw_slowdown", 0),
            "devfreq_cur_hz": devfreq_cur,
            "stress_pid": cuda_proc_pid,
        })

        prev_current_a = curr_a
        prev_time = t_curr
        time.sleep(0.02)

    return records


def run_contention_isolation(
    cuda_binary=None,
    duration_per_phase=15.0,
    num_trials=10,
    output_dir=None,
    arch_override=None,
    seed=None,
    victim_type="synthetic",
):
    if cuda_binary is None:
        cuda_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3" + _EXE)
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(SCRIPT_DIR), "..", "results", "contention_isolation"
        )
    os.makedirs(output_dir, exist_ok=True)

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    print("=" * 70)
    print(" Project GAIA: Contention-vs-Power Isolation Experiment")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s | Seed: {seed}")
    print(f" Victim: {victim_type}")
    print("=" * 70)

    if not os.path.exists(cuda_binary):
        compile_cuda_backend(out_binary=cuda_binary, arch_override=arch_override)

    assessor = GPUFaultAssessorV2(arch_override=arch_override)

    from realistic_victims import create_victim
    victim_fn = create_victim(victim_type)

    platform_info = collect_platform_info()
    platform_info["experiment"] = "contention_isolation"
    platform_info["seed"] = seed
    platform_info["victim_type"] = victim_type
    platform_info["num_trials"] = num_trials
    platform_info["duration_per_phase"] = duration_per_phase
    platform_info["conditions"] = [
        {"name": c[0], "mode": c[1], "freq_khz": c[2], "duty": c[3], "desc": c[4]}
        for c in CONDITIONS
    ]
    with open(os.path.join(output_dir, "platform_info.json"), "w") as f:
        json.dump(platform_info, f, indent=2, default=str)

    devfreq_path = assessor._devfreq_path
    has_devfreq = devfreq_path is not None

    if has_devfreq:
        orig_snap = devfreq_snapshot(devfreq_path)
        fixed_freq_hz = devfreq_pick_fixed_freq(devfreq_path)
        print(f"[clock] Original: governor={orig_snap['governor']}, "
              f"max={orig_snap['max_freq']}")
        print(f"[clock] Fixed frequency: {fixed_freq_hz} Hz "
              f"({fixed_freq_hz / 1e6:.0f} MHz)")
        clock_policies = ["default", "fixed"]
    else:
        orig_snap = None
        fixed_freq_hz = None
        clock_policies = ["default"]
        print("[clock] No devfreq path — running default clock only")

    # Build the full schedule: each condition × each clock policy × baseline + attack
    # For each (condition, clock_policy) we run:
    #   1. victim-only baseline (at that clock state)
    #   2. victim + attacker
    # This ensures clock-matched baselines.
    schedule_items = []
    for cond_name, mode, freq, duty, desc in CONDITIONS:
        for cpol in clock_policies:
            tag = f"{cond_name}__{cpol}"
            # Baseline at this clock policy
            schedule_items.append({
                "tag": f"{tag}__baseline",
                "condition": cond_name,
                "mode": None,
                "freq": 0,
                "duty": 0.0,
                "clock_policy": cpol,
                "is_attack": False,
            })
            # Attack (skip if baseline_idle — already have victim-only)
            if mode is not None:
                schedule_items.append({
                    "tag": f"{tag}__attack",
                    "condition": cond_name,
                    "mode": mode,
                    "freq": freq,
                    "duty": duty,
                    "clock_policy": cpol,
                    "is_attack": True,
                })

    print(f"\n[schedule] {len(schedule_items)} phases per trial "
          f"({len(CONDITIONS)} conditions × {len(clock_policies)} clock policies × baseline+attack)")

    # Warmup
    print("\n[+] Warmup (discarded)...")
    warmup_start = time.time()
    for _ in range(20):
        victim_fn()
    print(f"[+] Warmup done ({time.time() - warmup_start:.1f}s)\n")

    all_records = []

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'=' * 70}")
        print(f"  Trial {trial_idx}/{num_trials}")
        print(f"{'=' * 70}")

        trial_start = time.time()

        # Randomize phase order within each trial to avoid systematic drift
        trial_schedule = list(schedule_items)
        random.shuffle(trial_schedule)

        for item in trial_schedule:
            tag = item["tag"]
            cpol = item["clock_policy"]
            mode = item["mode"]

            print(f"  [{trial_idx}] {tag} ...")

            # Set clock policy
            if has_devfreq:
                try:
                    if cpol == "fixed":
                        devfreq_set_fixed(devfreq_path, fixed_freq_hz)
                    else:
                        devfreq_restore(devfreq_path, orig_snap)
                except (IOError, OSError) as e:
                    print(f"    [!] Clock set failed: {e}")

            # Cooldown between phases to reduce thermal carryover
            time.sleep(2.0)

            cuda_proc = None
            stress_pid = None
            if mode is not None:
                cuda_proc = _launch_stress(
                    cuda_binary, mode, item["freq"], item["duty"],
                    duration_per_phase
                )
                stress_pid = cuda_proc.pid

            records = _collect_phase(
                assessor, victim_fn, duration_per_phase, trial_start,
                phase_name=tag,
                is_attack=item["is_attack"],
                clock_policy=cpol,
                cuda_proc_pid=stress_pid,
            )

            if cuda_proc is not None:
                cuda_proc.terminate()
                try:
                    cuda_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    cuda_proc.kill()

            # Restore clock after each phase
            if has_devfreq:
                try:
                    devfreq_restore(devfreq_path, orig_snap)
                except (IOError, OSError):
                    pass

            for r in records:
                r["trial"] = trial_idx
                r["condition"] = item["condition"]
            all_records.extend(records)

            # Print phase summary
            if records:
                lat = np.mean([r["victim_total_latency_ms"] for r in records])
                pwr = np.mean([r["power_w"] for r in records])
                clk = np.mean([r["sm_clock_mhz"] for r in records])
                print(f"    Lat: {lat:.2f} ms | Power: {pwr:.2f} W | Clock: {clk:.0f} MHz")

    # Final clock restore
    if has_devfreq and orig_snap:
        try:
            devfreq_restore(devfreq_path, orig_snap)
        except (IOError, OSError):
            pass

    # Save raw data
    csv_path = os.path.join(output_dir, "contention_isolation.csv")
    if all_records:
        fieldnames = [
            "trial", "timestamp", "condition", "phase", "is_attack",
            "clock_policy", "power_w", "voltage_v", "current_a",
            "di_dt_amps_per_sec",
            "victim_total_latency_ms", "victim_gpu_compute_ms",
            "victim_launch_delay_ms", "victim_backend",
            "sm_clock_mhz", "gpu_temp_c", "gpu_util_pct", "mem_util_pct",
            "sw_power_cap", "hw_slowdown", "devfreq_cur_hz", "stress_pid",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[+] Raw data: {csv_path} ({len(all_records)} rows)")

    # Compute and save summary
    _print_isolation_summary(all_records, output_dir)

    return csv_path


def _print_isolation_summary(all_records, output_dir):
    """Compute per-condition means and the degradation decomposition."""
    if not all_records:
        print("[!] No records to summarize")
        return

    df = pd.DataFrame(all_records)

    summary_rows = []
    print("\n" + "=" * 90)
    print("  CONTENTION ISOLATION SUMMARY")
    print("=" * 90)
    print(f"  {'Condition':<28}{'Clock':<8}{'Type':<10}"
          f"{'Lat(ms)':<12}{'Power(W)':<12}{'Clock(MHz)':<12}"
          f"{'GPU%':<8}{'Mem%':<8}")
    print("  " + "-" * 86)

    for condition in df["condition"].unique():
        for cpol in df["clock_policy"].unique():
            for is_atk in [0, 1]:
                sub = df[(df["condition"] == condition) &
                         (df["clock_policy"] == cpol) &
                         (df["is_attack"] == is_atk)]
                if sub.empty:
                    continue

                trial_lats = sub.groupby("trial")["victim_total_latency_ms"].mean()
                trial_pows = sub.groupby("trial")["power_w"].mean()
                trial_clks = sub.groupby("trial")["sm_clock_mhz"].mean()
                trial_gpu = sub.groupby("trial")["gpu_util_pct"].mean()
                trial_mem = sub.groupby("trial")["mem_util_pct"].mean()

                row = {
                    "condition": condition,
                    "clock_policy": cpol,
                    "type": "attack" if is_atk else "baseline",
                    "lat_mean": trial_lats.mean(),
                    "lat_std": trial_lats.std() if len(trial_lats) > 1 else 0,
                    "pow_mean": trial_pows.mean(),
                    "pow_std": trial_pows.std() if len(trial_pows) > 1 else 0,
                    "clk_mean": trial_clks.mean(),
                    "gpu_util_mean": trial_gpu.mean(),
                    "mem_util_mean": trial_mem.mean(),
                    "n_trials": trial_lats.nunique(),
                }
                summary_rows.append(row)

                typ = "ATK" if is_atk else "base"
                gpu_str = f"{trial_gpu.mean():.0f}" if trial_gpu.notna().any() else "N/A"
                mem_str = f"{trial_mem.mean():.0f}" if trial_mem.notna().any() else "N/A"
                print(f"  {condition:<28}{cpol:<8}{typ:<10}"
                      f"{trial_lats.mean():<12.2f}{trial_pows.mean():<12.2f}"
                      f"{trial_clks.mean():<12.0f}{gpu_str:<8}{mem_str:<8}")

    # Compute degradation for each attack condition relative to its
    # clock-matched baseline
    print("\n  DEGRADATION ANALYSIS (attack vs clock-matched baseline)")
    print("  " + "-" * 86)
    print(f"  {'Condition':<28}{'Clock':<8}{'Degrad%':<12}"
          f"{'ΔPower(W)':<12}{'ΔGPU%':<10}{'ΔMem%':<10}")
    print("  " + "-" * 86)

    degradation_rows = []
    for condition in df["condition"].unique():
        if condition == "baseline_idle":
            continue
        for cpol in df["clock_policy"].unique():
            base_rows = [r for r in summary_rows
                         if r["condition"] == condition
                         and r["clock_policy"] == cpol
                         and r["type"] == "baseline"]
            atk_rows = [r for r in summary_rows
                        if r["condition"] == condition
                        and r["clock_policy"] == cpol
                        and r["type"] == "attack"]
            if not base_rows or not atk_rows:
                continue
            b = base_rows[0]
            a = atk_rows[0]
            if b["lat_mean"] > 0:
                degrad_pct = ((a["lat_mean"] - b["lat_mean"]) / b["lat_mean"]) * 100
            else:
                degrad_pct = 0.0
            delta_pow = a["pow_mean"] - b["pow_mean"]
            delta_gpu = (a["gpu_util_mean"] or 0) - (b["gpu_util_mean"] or 0)
            delta_mem = (a["mem_util_mean"] or 0) - (b["mem_util_mean"] or 0)

            degradation_rows.append({
                "condition": condition,
                "clock_policy": cpol,
                "degradation_pct": degrad_pct,
                "delta_power_w": delta_pow,
                "delta_gpu_util": delta_gpu,
                "delta_mem_util": delta_mem,
                "baseline_lat": b["lat_mean"],
                "attack_lat": a["lat_mean"],
                "baseline_power": b["pow_mean"],
                "attack_power": a["pow_mean"],
            })

            print(f"  {condition:<28}{cpol:<8}{degrad_pct:<+12.1f}"
                  f"{delta_pow:<+12.2f}{delta_gpu:<+10.1f}{delta_mem:<+10.1f}")

    # Pair analysis
    print("\n  PAIR 1: Same resource type, different power (compute vs oversubscribe)")
    for cpol in df["clock_policy"].unique():
        p1c = [d for d in degradation_rows
               if d["condition"] == "pair1_compute" and d["clock_policy"] == cpol]
        p1o = [d for d in degradation_rows
               if d["condition"] == "pair1_oversubscribe" and d["clock_policy"] == cpol]
        if p1c and p1o:
            diff = p1o[0]["degradation_pct"] - p1c[0]["degradation_pct"]
            pow_diff = p1o[0]["attack_power"] - p1c[0]["attack_power"]
            print(f"    [{cpol}] Oversubscribe draws {pow_diff:+.1f}W more, "
                  f"causes {diff:+.1f}% additional degradation")
            if abs(diff) > 5:
                print(f"    → Power-domain signal detected (>{5}% additional degradation)")
            else:
                print(f"    → No clear power-domain signal (<5% difference)")

    print("\n  PAIR 2: Matched power, different resource profile (cache vs compute)")
    for cpol in df["clock_policy"].unique():
        p2c = [d for d in degradation_rows
               if d["condition"] == "pair2_cache_full" and d["clock_policy"] == cpol]
        p2t = [d for d in degradation_rows
               if d["condition"] == "pair2_compute_tuned" and d["clock_policy"] == cpol]
        if p2c and p2t:
            pow_gap = abs(p2c[0]["attack_power"] - p2t[0]["attack_power"])
            deg_gap = p2c[0]["degradation_pct"] - p2t[0]["degradation_pct"]
            print(f"    [{cpol}] Power gap: {pow_gap:.1f}W | "
                  f"Degradation gap: {deg_gap:+.1f}%")
            if pow_gap < 5 and abs(deg_gap) > 10:
                print(f"    → Resource-profile effect dominates (matched power, different degradation)")
            elif pow_gap >= 5:
                print(f"    → Power not fully matched ({pow_gap:.1f}W gap); interpret with caution")

    print("\n  PAIR 3: DVFS isolation (default vs fixed clock)")
    if "fixed" in df["clock_policy"].unique().tolist():
        for cond in ["pair1_compute", "pair1_oversubscribe", "pair2_cache_full"]:
            dd = [d for d in degradation_rows
                  if d["condition"] == cond and d["clock_policy"] == "default"]
            df_ = [d for d in degradation_rows
                   if d["condition"] == cond and d["clock_policy"] == "fixed"]
            if dd and df_:
                diff = dd[0]["degradation_pct"] - df_[0]["degradation_pct"]
                print(f"    {cond}: default={dd[0]['degradation_pct']:+.1f}% "
                      f"fixed={df_[0]['degradation_pct']:+.1f}% "
                      f"(DVFS contribution: {diff:+.1f}%)")

    # Save summary CSV
    if summary_rows:
        import pandas as _pd
        sum_df = _pd.DataFrame(summary_rows)
        sum_path = os.path.join(output_dir, "contention_isolation_summary.csv")
        sum_df.to_csv(sum_path, index=False)
        print(f"\n[+] Summary: {sum_path}")

    if degradation_rows:
        import pandas as _pd
        deg_df = _pd.DataFrame(degradation_rows)
        deg_path = os.path.join(output_dir, "contention_isolation_degradation.csv")
        deg_df.to_csv(deg_path, index=False)
        print(f"[+] Degradation: {deg_path}")

    print("\n" + "=" * 90)
    print("  To run regression analysis on these results:")
    print("    degradation ~ power_w + gpu_util_pct + mem_util_pct")
    print("  Use the degradation CSV with statsmodels OLS or sklearn.")
    print("=" * 90 + "\n")


# Need pandas for the summary — import at module level for the summary function
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project GAIA: Contention-vs-Power Isolation Experiment"
    )
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration per phase in seconds (default: 15)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of recorded trials (default: 10)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: results/contention_isolation)")
    parser.add_argument("--arch", type=str, default=None,
                        help="GPU arch override, e.g. sm_110 for Thor")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--victim-type", type=str, default="synthetic",
                        choices=["synthetic", "training", "inference"],
                        help="Victim workload type (default: synthetic)")
    args = parser.parse_args()

    run_contention_isolation(
        duration_per_phase=args.duration,
        num_trials=args.trials,
        output_dir=args.out,
        arch_override=args.arch,
        seed=args.seed,
        victim_type=args.victim_type,
    )
