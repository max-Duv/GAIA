#!/usr/bin/env python3
"""
Project GAIA — H100 Power-Cap Oversubscription Sweep (Thrust T1)

Sweeps nvidia-smi --power-limit across a range of values and measures
whether the attack persists under each cap. Extends the Jetson devfreq
oversubscription experiment to discrete GPUs.

Conditions at each power cap:
  1. Victim alone (capped baseline)
  2. Victim + oversubscription attack (capped attack)

Usage:
    python h100_powercap_sweep.py --trials 3 --duration 15
    python h100_powercap_sweep.py --caps 300,400,500,600,700
"""

import os
import sys
import time
import csv
import json
import argparse
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "T1")
sys.path.insert(0, T1_DIR)
sys.path.insert(0, SCRIPT_DIR)

from h100_attack_transfer import (
    H100Telemetry, compile_attack_kernel, detect_gpu_arch,
    collect_h100_platform_info, _EXE
)
from realistic_victims import SyntheticMatmulVictim


def get_default_power_limit(gpu_idx=0):
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_idx}",
             "--query-gpu=power.default_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            val = r.stdout.strip()
            if val not in ("[N/A]", "N/A", ""):
                return float(val)
    except Exception:
        pass
    return 700.0


def set_power_limit(watts, gpu_idx=0):
    cmd = ["nvidia-smi", f"--id={gpu_idx}", "-pl", str(int(watts))]
    if sys.platform != "win32":
        cmd.insert(0, "sudo")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"    [!] Failed to set power limit to {watts}W: {r.stderr.strip()}")
        return False
    print(f"    [+] Power limit set to {int(watts)}W")
    return True


def restore_power_limit(default_w, gpu_idx=0):
    set_power_limit(default_w, gpu_idx)


def run_sweep(args):
    telemetry = H100Telemetry(gpu_idx=args.gpu)
    platform_info = collect_h100_platform_info(telemetry)
    gpu_name = platform_info.get("device_model", "unknown")
    default_limit = get_default_power_limit(args.gpu)

    caps = [int(c) for c in args.caps.split(",")]
    print(f"\n{'='*70}")
    print(f"  GAIA H100 POWER-CAP OVERSUBSCRIPTION SWEEP")
    print(f"  GPU: {gpu_name} | Default limit: {default_limit}W")
    print(f"  Caps: {caps}")
    print(f"  Trials: {args.trials} | Duration/phase: {args.duration}s")
    print(f"{'='*70}\n")

    attack_binary = compile_attack_kernel(args.arch)
    buf_size_mb = 64
    victim = SyntheticMatmulVictim(device_idx=args.gpu,
                                   matrix_size=args.matrix_size)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, f"results_powercap_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    platform_info["sweep_caps_w"] = caps
    platform_info["default_power_limit_w"] = default_limit
    with open(os.path.join(out_dir, "platform_info.json"), "w") as f:
        json.dump(platform_info, f, indent=2)

    fieldnames = [
        "trial", "timestamp", "phase", "is_attack", "category",
        "power_cap_w", "data_source",
        "power_w", "voltage_v", "current_a", "di_dt_amps_per_sec",
        "victim_kernel_ms", "victim_total_latency_ms",
        "sm_clock_mhz", "gpu_temp_c", "gpu_util_pct", "mem_util_pct",
        "sw_power_cap", "hw_slowdown", "throttle_bitmask",
    ]

    all_records = []

    try:
        for cap_w in caps:
            print(f"\n{'='*60}")
            print(f"  POWER CAP: {cap_w}W")
            print(f"{'='*60}")

            if not set_power_limit(cap_w, args.gpu):
                print(f"    [!] Skipping cap {cap_w}W (failed to set)")
                continue
            time.sleep(2)

            for trial in range(1, args.trials + 1):
                for phase_name, run_stress in [("capped_baseline", False),
                                                ("capped_attack", True)]:
                    print(f"  [{trial}] {phase_name} @ {cap_w}W")

                    cuda_proc = None
                    if run_stress:
                        stress_cmd = [
                            attack_binary,
                            "--mode", "oversubscribe",
                            "--freq", "10.0",
                            "--duty", "0.5",
                            "--duration", str(int(args.duration + 5)),
                            "--streams", "8",
                            "--bufsize", str(buf_size_mb),
                        ]
                        cuda_proc = subprocess.Popen(
                            stress_cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
                        time.sleep(1.0)

                    prev_current = 0.0
                    prev_time = time.time()
                    phase_start = time.time()

                    while (time.time() - phase_start) < args.duration:
                        v_res = victim()
                        t_now = time.time()
                        dt = max(t_now - prev_time, 0.001)
                        hw = telemetry.read()

                        power_w = hw["power_w"]
                        voltage_v = hw.get("voltage_v", 12.0)
                        if voltage_v <= 0:
                            voltage_v = 12.0
                        current_a = power_w / voltage_v
                        di_dt = (current_a - prev_current) / dt

                        all_records.append({
                            "trial": trial,
                            "timestamp": round(t_now - phase_start, 4),
                            "phase": phase_name,
                            "is_attack": int(run_stress),
                            "category": "sweep_powercap",
                            "power_cap_w": cap_w,
                            "data_source": hw["provenance"],
                            "power_w": round(power_w, 3),
                            "voltage_v": round(voltage_v, 5),
                            "current_a": round(current_a, 4),
                            "di_dt_amps_per_sec": round(di_dt, 4),
                            "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
                            "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                            "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                            "gpu_temp_c": hw.get("gpu_temp_c", 0),
                            "gpu_util_pct": hw.get("gpu_util_pct", 0),
                            "mem_util_pct": hw.get("mem_util_pct", 0),
                            "sw_power_cap": hw.get("sw_power_cap", 0),
                            "hw_slowdown": hw.get("hw_slowdown", 0),
                            "throttle_bitmask": hw.get("throttle_bitmask", 0),
                        })
                        prev_current = current_a
                        prev_time = t_now
                        time.sleep(0.02)

                    if cuda_proc is not None:
                        cuda_proc.terminate()
                        try:
                            cuda_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            cuda_proc.kill()

                    time.sleep(2)

    finally:
        print(f"\n[+] Restoring default power limit ({default_limit}W)")
        restore_power_limit(default_limit, args.gpu)

    csv_path = os.path.join(out_dir, "h100_powercap_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED: {csv_path}")
    print(f"{'='*70}")

    # Summary per cap level
    print(f"\n{'='*70}")
    print(f"  POWER-CAP SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Cap(W)':>7s}  {'Phase':>18s}  {'Power(W)':>12s}  {'Latency(ms)':>14s}  {'Throttle':>8s}")
    for cap_w in caps:
        for phase_name in ["capped_baseline", "capped_attack"]:
            rows = [r for r in all_records
                    if r["power_cap_w"] == cap_w and r["phase"] == phase_name]
            if not rows:
                continue
            pows = [r["power_w"] for r in rows]
            lats = [r["victim_total_latency_ms"] for r in rows]
            swcap = sum(r["sw_power_cap"] for r in rows)
            print(f"  {cap_w:7d}  {phase_name:>18s}  "
                  f"{np.mean(pows):5.1f}±{np.std(pows):4.1f}  "
                  f"{np.mean(lats):7.2f}±{np.std(lats):5.2f}  "
                  f"{swcap:8d}")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA H100 Power-Cap Sweep")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--caps", type=str, default="300,400,500,600,700")
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()
    run_sweep(args)
