"""
Project GAIA: Filler Kernel Defense Experiment
===============================================
Addresses proposal commitment: T2 defense mechanism — "filler kernels."

Hypothesis: A lightweight filler kernel that proactively occupies idle SM
capacity reduces the attacker's available scheduling slots, dampening
adversarial power spikes and reducing the attack's power amplitude.

Conditions (5 trials each, 15s phases):
  A. victim_alone            — baseline power/latency
  B. victim_attacker         — undefended attack (control)
  C. victim_attacker_fill25  — attack + filler at 25% intensity
  D. victim_attacker_fill50  — attack + filler at 50% intensity
  E. victim_filler_only25    — filler overhead (no attack)
  F. victim_filler_only50    — filler overhead (no attack)

Metrics:
  - Attack power amplitude reduction (W)
  - Victim latency overhead from filler (ms)
  - Net defense benefit (power reduction vs latency cost)

Usage (on Jetson via SSH):
    sudo python3 filler_kernel_experiment.py --trials 5 --duration 15
"""

import os
import sys
import time
import json
import random
import subprocess
import csv
import argparse
import signal
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "T1")
CODE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "code")
sys.path.insert(0, T1_DIR)
sys.path.insert(0, CODE_DIR)

from gpu_fault_assessor_v2 import (
    GPUFaultAssessorV2, CONDITION_LABELS, collect_platform_info,
    _cuda_subprocess_env,
)
from controlled_3phase_runner_v2 import (
    compile_cuda_backend, detect_gpu_arch,
    _read_devfreq, _write_devfreq, devfreq_snapshot, devfreq_restore,
)
from filler_kernel_wrapper import FillerKernel, compile_filler_kernel

_EXE = ".exe" if sys.platform == "win32" else ""

CONDITIONS = [
    {"name": "victim_alone",            "attacker": False, "filler_pct": 0},
    {"name": "victim_attacker",         "attacker": True,  "filler_pct": 0},
    {"name": "victim_attacker_fill25",  "attacker": True,  "filler_pct": 25},
    {"name": "victim_attacker_fill50",  "attacker": True,  "filler_pct": 50},
    {"name": "victim_filler_only25",    "attacker": False, "filler_pct": 25},
    {"name": "victim_filler_only50",    "attacker": False, "filler_pct": 50},
]


def launch_attacker(cuda_binary, duration_s):
    cmd = [
        cuda_binary,
        "--mode", "oversubscribe",
        "--freq", "0.001",
        "--duty", "1.0",
        "--duration", str(duration_s + 3.0),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=_cuda_subprocess_env(),
    )
    return proc


def kill_process(proc, timeout=5):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def run_phase(assessor, victim_fn, condition, cuda_binary, duration_s, trial_num):
    """Run one condition phase. Returns list of per-sample records."""
    cond_name = condition["name"]
    run_attacker = condition["attacker"]
    filler_pct = condition["filler_pct"]

    print(f"  [T{trial_num}] Phase '{cond_name}' "
          f"(attacker={run_attacker}, filler={filler_pct}%)...")

    records = []
    attacker_proc = None
    filler = None
    prev_current_a = 0.0
    prev_time = time.time()

    try:
        # Start filler if needed
        if filler_pct > 0:
            filler = FillerKernel(intensity_pct=filler_pct, duty=0.5)
            filler.start(duration_s=0)
            time.sleep(0.5)
            print(f"    [+] Filler started (pid={filler.pid}, {filler_pct}%)")

        # Start attacker if needed
        if run_attacker:
            attacker_proc = launch_attacker(cuda_binary, duration_s)
            time.sleep(0.5)
            print(f"    [+] Attacker started (pid={attacker_proc.pid})")

        # Telemetry + victim loop
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

            rec = {
                "timestamp": t_curr,
                "elapsed_s": t_curr - phase_start,
                "trial": trial_num,
                "condition": cond_name,
                "is_attack": int(run_attacker),
                "filler_pct": filler_pct,
                "total_power_w": p_w,
                "voltage_v": v_meas,
                "current_a": curr_a,
                "di_dt": di_dt,
                "total_latency_ms": v_res["total_latency_ms"],
                "gpu_compute_ms": v_res["gpu_compute_ms"],
                "gpu_clock_mhz": hw.get("sm_clock_mhz", 0),
                "gpu_temp_c": hw.get("gpu_temp_c", 0),
                "gpu_util_pct": hw.get("gpu_util_pct", 0),
                "sw_power_cap": hw.get("sw_power_cap", 0),
                "hw_slowdown": hw.get("hw_slowdown", 0),
            }
            records.append(rec)
            prev_current_a = curr_a
            prev_time = t_curr

    finally:
        # Clean up
        if attacker_proc:
            kill_process(attacker_proc)
            print(f"    [-] Attacker stopped")
        if filler:
            filler.stop()
            print(f"    [-] Filler stopped")

    n = len(records)
    if n > 0:
        mean_pw = np.mean([r["total_power_w"] for r in records])
        mean_lat = np.mean([r["total_latency_ms"] for r in records])
        print(f"    => {n} samples, mean power={mean_pw:.1f}W, "
              f"mean latency={mean_lat:.2f}ms")

    return records


def run_experiment(args):
    print("=" * 70)
    print("Project GAIA: Filler Kernel Defense Experiment")
    print("=" * 70)

    # Compile backends
    arch = detect_gpu_arch()
    compile_cuda_backend(arch_override=arch)
    cuda_binary = os.path.join(T1_DIR, "gpu_power_poc_v3" + _EXE)

    filler_binary = os.path.join(CODE_DIR, "filler_kernel" + _EXE)
    if not os.path.exists(filler_binary):
        print("[setup] Compiling filler kernel...")
        compile_filler_kernel(arch=arch)

    assessor = GPUFaultAssessorV2()
    victim_fn = assessor.run_victim_workload_iteration
    platform_info = collect_platform_info()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, f"results_filler_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "platform_info.json"), "w") as f:
        json.dump(platform_info, f, indent=2, default=str)

    all_records = []

    for trial in range(1, args.trials + 1):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial}/{args.trials}")
        print(f"{'='*50}")

        # Randomize condition order per trial
        trial_conditions = CONDITIONS.copy()
        random.shuffle(trial_conditions)

        for cond in trial_conditions:
            # Cool-down between phases
            print(f"  [cooldown] Waiting {args.cooldown}s...")
            time.sleep(args.cooldown)

            records = run_phase(
                assessor, victim_fn, cond, cuda_binary,
                args.duration, trial,
            )
            all_records.extend(records)

    # Write raw CSV
    csv_path = os.path.join(out_dir, "filler_defense.csv")
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[output] Raw CSV: {csv_path} ({len(all_records)} rows)")

    # Compute summary statistics
    print_summary(all_records, out_dir)

    print(f"\n[done] Results in: {out_dir}")
    return out_dir


def print_summary(records, out_dir):
    if not records:
        return

    print("\n" + "=" * 70)
    print("SUMMARY: Filler Kernel Defense")
    print("=" * 70)

    summary_rows = []
    for cond in CONDITIONS:
        cname = cond["name"]
        cond_recs = [r for r in records if r["condition"] == cname]
        if not cond_recs:
            continue
        powers = [r["total_power_w"] for r in cond_recs]
        lats = [r["total_latency_ms"] for r in cond_recs]
        row = {
            "condition": cname,
            "is_attack": cond["attacker"],
            "filler_pct": cond["filler_pct"],
            "n_samples": len(cond_recs),
            "mean_power_w": np.mean(powers),
            "std_power_w": np.std(powers),
            "mean_latency_ms": np.mean(lats),
            "std_latency_ms": np.std(lats),
        }
        summary_rows.append(row)
        print(f"  {cname:30s}: power={row['mean_power_w']:6.1f} +/- "
              f"{row['std_power_w']:5.1f}W, latency={row['mean_latency_ms']:6.2f} "
              f"+/- {row['std_latency_ms']:5.2f}ms  (n={row['n_samples']})")

    # Compute defense metrics
    baseline = next((r for r in summary_rows
                     if r["condition"] == "victim_alone"), None)
    no_filler = next((r for r in summary_rows
                      if r["condition"] == "victim_attacker"), None)
    fill25 = next((r for r in summary_rows
                   if r["condition"] == "victim_attacker_fill25"), None)
    fill50 = next((r for r in summary_rows
                   if r["condition"] == "victim_attacker_fill50"), None)

    if baseline and no_filler:
        attack_overhead = no_filler["mean_power_w"] - baseline["mean_power_w"]
        print(f"\n  Attack power overhead (no filler): +{attack_overhead:.1f}W")

        for label, row in [("25% filler", fill25), ("50% filler", fill50)]:
            if row:
                defended_overhead = row["mean_power_w"] - baseline["mean_power_w"]
                reduction = attack_overhead - defended_overhead
                lat_cost = row["mean_latency_ms"] - no_filler["mean_latency_ms"]
                print(f"  {label}: power overhead={defended_overhead:+.1f}W "
                      f"(reduced by {reduction:.1f}W), "
                      f"latency cost={lat_cost:+.2f}ms")

    # Write summary CSV
    if summary_rows:
        csv_path = os.path.join(out_dir, "filler_defense_summary.csv")
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[output] Summary CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAIA Filler Kernel Defense Experiment")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials (default: 5)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Phase duration in seconds (default: 15)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Cooldown between phases (default: 5)")
    args = parser.parse_args()

    run_experiment(args)
