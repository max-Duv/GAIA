#!/usr/bin/env python3
"""
Project GAIA — H100 MIG Power Isolation Test (Thrust T1)

Tests whether NVIDIA Multi-Instance GPU (MIG) partitioning isolates
tenants from power-domain attacks. Creates two MIG instances, runs a
victim workload in one and an attacker in the other, and measures
whether power interference crosses MIG boundaries.

Requires: root/sudo, MIG-capable GPU (A100/H100), no other MIG users.

Experiment design:
  Phase 1: Victim alone in MIG instance 0 (baseline)
  Phase 2: Victim in MIG-0 + attacker in MIG-1 (cross-partition attack)
  Phase 3: Victim alone again (recovery baseline)

Usage:
    sudo python h100_mig_isolation.py --trials 5 --duration 20
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


# ---------------------------------------------------------------------------
# MIG management helpers
# ---------------------------------------------------------------------------

def mig_enabled(gpu_idx=0):
    r = subprocess.run(
        ["nvidia-smi", f"--id={gpu_idx}", "--query-gpu=mig.mode.current",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10)
    return r.returncode == 0 and "Enabled" in r.stdout


def enable_mig(gpu_idx=0):
    print("[mig] Enabling MIG mode...")
    cmd = ["sudo", "nvidia-smi", f"--id={gpu_idx}", "-mig", "1"]
    if sys.platform == "win32":
        cmd = cmd[1:]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to enable MIG: {r.stderr}")
    print("[mig] MIG enabled (may require GPU reset)")


def disable_mig(gpu_idx=0):
    print("[mig] Disabling MIG mode...")
    cmd = ["sudo", "nvidia-smi", f"--id={gpu_idx}", "-mig", "0"]
    if sys.platform == "win32":
        cmd = cmd[1:]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"[mig] WARNING: Failed to disable MIG: {r.stderr}")
    else:
        print("[mig] MIG disabled")


def destroy_all_mig_instances(gpu_idx=0):
    print("[mig] Destroying existing MIG instances...")
    prefix = ["sudo"] if sys.platform != "win32" else []
    subprocess.run(prefix + ["nvidia-smi", "mig", "-dci", "-f",
                              f"--id={gpu_idx}"],
                   capture_output=True, text=True, timeout=15)
    subprocess.run(prefix + ["nvidia-smi", "mig", "-dgi", "-f",
                              f"--id={gpu_idx}"],
                   capture_output=True, text=True, timeout=15)
    time.sleep(1)


def create_mig_instances(gpu_idx=0, profile="9,9"):
    """Create two equal MIG GPU instances and their compute instances.

    Default profile '9,9' creates two 3g.40gb instances on H100.
    For A100 80GB, use '9,9' (2x 3g.40gb) or '19,19' (2x 4g.40gb).
    """
    prefix = ["sudo"] if sys.platform != "win32" else []

    print(f"[mig] Creating GPU instances (profile: {profile})...")
    r = subprocess.run(
        prefix + ["nvidia-smi", "mig", "-cgi", profile,
                  f"--id={gpu_idx}"],
        capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create GPU instances: {r.stderr}")

    print("[mig] Creating compute instances...")
    r = subprocess.run(
        prefix + ["nvidia-smi", "mig", "-cci",
                  f"--id={gpu_idx}"],
        capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create compute instances: {r.stderr}")

    time.sleep(1)
    print("[mig] MIG instances created successfully")


def list_mig_devices():
    """Return a list of MIG device UUIDs (MIG-GPU-xxx format)."""
    r = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return []

    devices = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if "MIG" in line and "UUID:" in line:
            uuid_start = line.index("UUID:") + 5
            uuid_end = line.index(")", uuid_start)
            uuid = line[uuid_start:uuid_end].strip()
            devices.append(uuid)
    return devices


def get_mig_cuda_devices():
    """Map MIG UUIDs to CUDA_VISIBLE_DEVICES indices.

    When MIG is enabled, each MIG instance gets its own CUDA device index.
    Returns list of (index, uuid) tuples.
    """
    mig_uuids = list_mig_devices()
    if len(mig_uuids) < 2:
        return []
    return [(i, uuid) for i, uuid in enumerate(mig_uuids)]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_mig_experiment(args):
    telemetry = H100Telemetry(gpu_idx=args.gpu)
    platform_info = collect_h100_platform_info(telemetry)
    gpu_name = platform_info.get("device_model", "unknown")

    print(f"\n{'='*70}")
    print(f"  GAIA H100 MIG POWER ISOLATION TEST")
    print(f"  GPU: {gpu_name}")
    print(f"  Trials: {args.trials} | Duration/phase: {args.duration}s")
    print(f"{'='*70}\n")

    attack_binary = compile_attack_kernel(args.arch)
    victim_src = os.path.join(T1_DIR, "gpu_victim_kernel.cu")
    victim_binary = os.path.join(SCRIPT_DIR, "gpu_victim_mig" + _EXE)

    arch = args.arch or detect_gpu_arch()
    r = subprocess.run(
        ["nvcc", "-O3", f"-arch={arch}", victim_src, "-o", victim_binary, "-lcuda"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Victim kernel compile failed: {r.stderr}")

    was_mig = mig_enabled(args.gpu)
    mig_setup_done = False

    try:
        if not was_mig:
            enable_mig(args.gpu)
            # MIG enable may require nvidia-smi reset
            subprocess.run(
                ["sudo", "nvidia-smi", f"--id={args.gpu}", "-r"],
                capture_output=True, text=True, timeout=30)
            time.sleep(3)

        destroy_all_mig_instances(args.gpu)
        create_mig_instances(args.gpu, profile=args.mig_profile)
        mig_setup_done = True

        mig_devices = get_mig_cuda_devices()
        if len(mig_devices) < 2:
            mig_uuids = list_mig_devices()
            print(f"[mig] Found MIG UUIDs: {mig_uuids}")
            raise RuntimeError(
                f"Need at least 2 MIG instances, found {len(mig_devices)}. "
                f"Try a different --mig-profile (e.g., '9,9' for H100, '19,19' for A100).")

        victim_uuid = mig_devices[0][1]
        attacker_uuid = mig_devices[1][1]
        print(f"[mig] Victim instance:   {victim_uuid}")
        print(f"[mig] Attacker instance:  {attacker_uuid}")

        # --- Run experiment ---
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(SCRIPT_DIR, f"results_mig_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        platform_info["mig_victim_uuid"] = victim_uuid
        platform_info["mig_attacker_uuid"] = attacker_uuid
        platform_info["mig_profile"] = args.mig_profile
        with open(os.path.join(out_dir, "platform_info.json"), "w") as f:
            json.dump(platform_info, f, indent=2)

        conditions = [
            {"name": "mig_victim_alone",   "run_attack": False, "category": "mig_baseline"},
            {"name": "mig_cross_attack",   "run_attack": True,  "category": "mig_attack"},
            {"name": "mig_recovery",       "run_attack": False, "category": "mig_baseline"},
        ]

        fieldnames = [
            "trial", "timestamp", "phase", "is_attack", "category",
            "data_source", "power_w", "gpu_temp_c",
            "gpu_util_pct", "mem_util_pct", "sm_clock_mhz",
            "sw_power_cap", "hw_slowdown", "throttle_bitmask",
            "victim_latency_ms",
        ]
        all_records = []

        for trial in range(1, args.trials + 1):
            print(f"\n{'='*60}")
            print(f"  Trial {trial}/{args.trials}")
            print(f"{'='*60}")

            for cond in conditions:
                p_name = cond["name"]
                print(f"\n  [{trial}] {p_name}")

                # Launch victim in MIG instance 0
                victim_env = os.environ.copy()
                victim_env["CUDA_VISIBLE_DEVICES"] = victim_uuid
                victim_cmd = [
                    victim_binary,
                    "--reps", str(999999),
                    "--elements", "16777216",
                    "--inner-iters", "64",
                ]
                victim_proc = subprocess.Popen(
                    victim_cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                    env=victim_env)

                # Optionally launch attacker in MIG instance 1
                attack_proc = None
                if cond["run_attack"]:
                    attack_env = os.environ.copy()
                    attack_env["CUDA_VISIBLE_DEVICES"] = attacker_uuid
                    attack_cmd = [
                        attack_binary,
                        "--mode", "oversubscribe",
                        "--freq", "10.0",
                        "--duty", "0.5",
                        "--duration", str(int(args.duration + 10)),
                        "--streams", "8",
                        "--bufsize", "32",
                    ]
                    attack_proc = subprocess.Popen(
                        attack_cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        env=attack_env)
                    print(f"    [+] Attacker launched in MIG-1 (PID {attack_proc.pid})")
                    time.sleep(1.0)

                phase_start = time.time()
                victim_latencies = []

                while (time.time() - phase_start) < args.duration:
                    hw = telemetry.read()

                    # Parse victim latency from stdout
                    lat_ms = 0.0
                    if victim_proc.poll() is None:
                        try:
                            line = victim_proc.stdout.readline()
                            if line and "VICTIM_LATENCY_MS:" in line:
                                lat_ms = float(line.split(":")[1].strip())
                                victim_latencies.append(lat_ms)
                        except Exception:
                            pass

                    all_records.append({
                        "trial": trial,
                        "timestamp": round(time.time() - phase_start, 4),
                        "phase": p_name,
                        "is_attack": int(cond["run_attack"]),
                        "category": cond["category"],
                        "data_source": hw["provenance"],
                        "power_w": round(hw["power_w"], 3),
                        "gpu_temp_c": hw.get("gpu_temp_c", 0),
                        "gpu_util_pct": hw.get("gpu_util_pct", 0),
                        "mem_util_pct": hw.get("mem_util_pct", 0),
                        "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                        "sw_power_cap": hw.get("sw_power_cap", 0),
                        "hw_slowdown": hw.get("hw_slowdown", 0),
                        "throttle_bitmask": hw.get("throttle_bitmask", 0),
                        "victim_latency_ms": round(lat_ms, 4),
                    })
                    time.sleep(0.05)

                # Cleanup
                victim_proc.terminate()
                try:
                    victim_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    victim_proc.kill()

                if attack_proc is not None:
                    attack_proc.terminate()
                    try:
                        attack_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        attack_proc.kill()

                if victim_latencies:
                    mean_lat = np.mean(victim_latencies)
                    print(f"    Victim latency: {mean_lat:.2f} ms "
                          f"(n={len(victim_latencies)})")

                time.sleep(3)

        # Save results
        csv_path = os.path.join(out_dir, "h100_mig_isolation.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        print(f"\n{'='*70}")
        print(f"  RESULTS SAVED: {csv_path}")
        print(f"{'='*70}")

        # Summary
        print(f"\n{'='*70}")
        print(f"  MIG ISOLATION SUMMARY")
        print(f"{'='*70}")
        for cond in conditions:
            rows = [r for r in all_records if r["phase"] == cond["name"]]
            if not rows:
                continue
            pows = [r["power_w"] for r in rows]
            lats = [r["victim_latency_ms"] for r in rows if r["victim_latency_ms"] > 0]
            lat_str = f"{np.mean(lats):.2f}±{np.std(lats):.2f}" if lats else "N/A"
            print(f"  {cond['name']:25s}  "
                  f"power={np.mean(pows):.1f}±{np.std(pows):.1f} W  "
                  f"victim_lat={lat_str} ms")

        # Key finding
        baseline_rows = [r for r in all_records if r["phase"] == "mig_victim_alone"]
        attack_rows = [r for r in all_records if r["phase"] == "mig_cross_attack"]
        if baseline_rows and attack_rows:
            bp = np.mean([r["power_w"] for r in baseline_rows])
            ap = np.mean([r["power_w"] for r in attack_rows])
            bl = [r["victim_latency_ms"] for r in baseline_rows if r["victim_latency_ms"] > 0]
            al = [r["victim_latency_ms"] for r in attack_rows if r["victim_latency_ms"] > 0]
            print(f"\n  Power elevation: {ap:.1f}W vs {bp:.1f}W "
                  f"(+{(ap-bp)/bp*100:.1f}%)")
            if bl and al:
                print(f"  Latency impact:  {np.mean(al):.2f} vs {np.mean(bl):.2f} ms "
                      f"(+{(np.mean(al)-np.mean(bl))/np.mean(bl)*100:.1f}%)")

        return csv_path

    finally:
        if mig_setup_done:
            print("\n[mig] Cleaning up MIG instances...")
            destroy_all_mig_instances(args.gpu)
        if not was_mig:
            disable_mig(args.gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA H100 MIG Isolation Test")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--mig-profile", type=str, default="9,9",
                        help="MIG GPU instance profile IDs (default: '9,9' = 2x 3g.40gb)")
    args = parser.parse_args()
    run_mig_experiment(args)
