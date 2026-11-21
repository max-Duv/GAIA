#!/usr/bin/env python3
"""
Project GAIA — H100 End-to-End Closed-Loop Pipeline (Thrust T3)

Integrates detection (T2) with mitigation on a discrete data-center GPU:
  Phase 1: Victim alone (baseline)
  Phase 2: Victim + attack, unmitigated (attack baseline)
  Phase 3: Victim + attack, power-cap mitigation via nvidia-smi -pl
  Phase 4: Victim + attack, process termination mitigation
  Phase 5: Recovery baseline (post-termination)

Mitigation mechanisms (discrete GPU equivalents of Jetson devfreq):
  - nvidia-smi --power-limit: software power cap (analogous to clock capping)
  - Process identification + SIGTERM/SIGKILL: adversarial kernel termination
  - Blinded PID attribution via nvidia-smi compute process listing

Usage:
    sudo python h100_closed_loop_pipeline.py --trials 5 --duration 15
"""

import os
import sys
import time
import csv
import json
import argparse
import subprocess
import signal
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "T1")
sys.path.insert(0, T1_DIR)
sys.path.insert(0, SCRIPT_DIR)

from h100_attack_transfer import (
    H100Telemetry, compile_attack_kernel, detect_gpu_arch,
    collect_h100_platform_info, _EXE
)
from h100_dcgm_detector import H100DetectorTelemetry, H100AnomalyDetector
from realistic_victims import SyntheticMatmulVictim


# ---------------------------------------------------------------------------
# Power-limit mitigation (nvidia-smi -pl)
# ---------------------------------------------------------------------------

def get_default_power_limit(gpu_idx=0):
    for field in ["power.default_limit", "power.limit"]:
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={gpu_idx}",
                 f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                val = r.stdout.strip()
                if val not in ("[N/A]", "N/A", ""):
                    return float(val)
        except Exception:
            pass
    return 700.0


def apply_power_cap(watts, gpu_idx=0):
    cmd = ["nvidia-smi", f"--id={gpu_idx}", "-pl", str(int(watts))]
    if sys.platform != "win32":
        cmd.insert(0, "sudo")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        print(f"    [!] Power cap failed: {r.stderr.strip()}")
        return False
    print(f"    [+] Power cap applied: {int(watts)}W")
    return True


def restore_power_limit(default_w, gpu_idx=0):
    apply_power_cap(default_w, gpu_idx)
    print(f"    [+] Power limit restored: {int(default_w)}W")


# ---------------------------------------------------------------------------
# PID attribution via nvidia-smi
# ---------------------------------------------------------------------------

def get_gpu_compute_pids(gpu_idx=0):
    """List PIDs of CUDA processes on the target GPU."""
    cmd = [
        "nvidia-smi", f"--id={gpu_idx}",
        "--query-compute-apps=pid,name,used_memory",
        "--format=csv,noheader,nounits"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return []
        pids = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1:
                try:
                    pids.append({
                        "pid": int(parts[0]),
                        "name": parts[1] if len(parts) > 1 else "unknown",
                        "mem_mb": float(parts[2]) if len(parts) > 2 else 0.0,
                    })
                except (ValueError, IndexError):
                    pass
        return pids
    except Exception:
        return []


def identify_adversarial_pid(gpu_idx=0, known_victim_pids=None):
    """Blinded attribution: flag PIDs not in the known-good set."""
    if known_victim_pids is None:
        known_victim_pids = set()
    all_pids = get_gpu_compute_pids(gpu_idx)
    suspects = [p for p in all_pids if p["pid"] not in known_victim_pids]
    return suspects


def terminate_process(pid, escalate=True):
    """Terminate a process by PID. Try SIGTERM first, escalate to SIGKILL."""
    method = "SIGTERM"
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1.0)
        try:
            os.kill(pid, 0)
            if escalate:
                os.kill(pid, signal.SIGKILL)
                method = "SIGKILL"
        except OSError:
            pass
        return True, method
    except OSError as e:
        print(f"    [!] Failed to terminate PID {pid}: {e}")
        return False, "failed"


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASES = [
    {
        "name": "victim_alone",
        "is_attack": False,
        "run_stress": False,
        "mitigate": False,
        "terminate": False,
        "category": "baseline",
    },
    {
        "name": "attack_unmitigated",
        "is_attack": True,
        "run_stress": True,
        "mitigate": False,
        "terminate": False,
        "category": "t3_attack",
    },
    {
        "name": "attack_powercap_mitigated",
        "is_attack": True,
        "run_stress": True,
        "mitigate": True,
        "terminate": False,
        "category": "t3_attack",
    },
    {
        "name": "attack_terminated",
        "is_attack": True,
        "run_stress": True,
        "mitigate": False,
        "terminate": True,
        "category": "t3_attack",
    },
    {
        "name": "recovery_baseline",
        "is_attack": False,
        "run_stress": False,
        "mitigate": False,
        "terminate": False,
        "category": "t3_baseline",
    },
]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_pipeline(args):
    tel_raw = H100Telemetry(gpu_idx=args.gpu)
    tel_det = H100DetectorTelemetry(gpu_idx=args.gpu)
    detector = H100AnomalyDetector(threshold_w=args.threshold, buffer_len=256)
    platform_info = collect_h100_platform_info(tel_raw)
    gpu_name = platform_info.get("device_model", "unknown")
    default_limit = get_default_power_limit(args.gpu)
    cap_limit = default_limit * args.cap_ratio

    print(f"\n{'='*70}")
    print(f"  GAIA H100 CLOSED-LOOP MITIGATION PIPELINE")
    print(f"  GPU: {gpu_name}")
    print(f"  Default PL: {default_limit}W | Cap target: {cap_limit:.0f}W")
    print(f"  Detection threshold: {args.threshold}W")
    print(f"  Trials: {args.trials} | Duration/phase: {args.duration}s")
    print(f"{'='*70}\n")

    attack_binary = compile_attack_kernel(args.arch)
    buf_size_mb = 64

    victim = SyntheticMatmulVictim(device_idx=args.gpu,
                                   matrix_size=args.matrix_size)
    victim_pid = os.getpid()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, f"results_pipeline_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    platform_info["default_power_limit_w"] = default_limit
    platform_info["cap_target_w"] = cap_limit
    platform_info["detection_threshold_w"] = args.threshold
    with open(os.path.join(out_dir, "platform_info.json"), "w") as f:
        json.dump(platform_info, f, indent=2)

    fieldnames = [
        "trial", "timestamp", "phase", "is_attack", "category",
        "data_source",
        "power_w", "voltage_v", "current_a", "di_dt_amps_per_sec",
        "victim_kernel_ms", "victim_total_latency_ms",
        "anomaly_detected", "mitigation_active", "terminated",
        "sm_clock_mhz", "gpu_temp_c", "gpu_util_pct", "mem_util_pct",
        "sw_power_cap", "hw_slowdown", "throttle_bitmask",
        "adversary_pid", "mitigation_latency_ms",
    ]

    all_records = []

    try:
        for trial in range(1, args.trials + 1):
            print(f"\n{'='*60}")
            print(f"  Trial {trial}/{args.trials}")
            print(f"{'='*60}")

            for phase in PHASES:
                p_name = phase["name"]
                print(f"\n  [{trial}] Phase: {p_name}")

                detector.power_buffer.clear()
                detector.time_buffer.clear()

                cuda_proc = None
                mitigation_active = False
                termination_done = False
                adversary_pid = None
                mitigation_latency_ms = None

                if phase["run_stress"]:
                    stress_cmd = [
                        attack_binary,
                        "--mode", "oversubscribe",
                        "--freq", "10.0",
                        "--duty", "0.5",
                        "--duration", str(int(args.duration + 10)),
                        "--streams", "8",
                        "--bufsize", str(buf_size_mb),
                    ]
                    cuda_proc = subprocess.Popen(
                        stress_cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True)
                    adversary_pid = cuda_proc.pid
                    print(f"    [+] Stress launched (PID {adversary_pid})")
                    time.sleep(1.0)

                prev_current = 0.0
                prev_time = time.time()
                phase_start = time.time()
                global_start = phase_start

                while (time.time() - phase_start) < args.duration:
                    v_res = victim()
                    t_now = time.time()
                    dt = max(t_now - prev_time, 0.001)

                    hw = tel_raw.read()
                    power_w = hw["power_w"]
                    voltage_v = 12.0
                    current_a = power_w / voltage_v
                    di_dt = (current_a - prev_current) / dt

                    hw_det = tel_det.read()
                    features = detector.update(t_now - global_start, hw_det)
                    is_anom, mean_p = detector.detect_threshold()

                    # Power-cap mitigation
                    if phase["mitigate"] and is_anom and not mitigation_active:
                        mit_start = time.time()
                        print(f"    [!] ANOMALY: {mean_p:.0f}W > {args.threshold:.0f}W "
                              f"— applying power cap ({cap_limit:.0f}W)")
                        ok = apply_power_cap(cap_limit, args.gpu)
                        mitigation_latency_ms = (time.time() - mit_start) * 1000
                        if ok:
                            mitigation_active = True
                            print(f"    [+] Mitigation applied in "
                                  f"{mitigation_latency_ms:.1f}ms")

                    # Termination mitigation
                    if phase["terminate"] and is_anom and not termination_done:
                        mit_start = time.time()
                        print(f"    [!] ANOMALY: {mean_p:.0f}W > {args.threshold:.0f}W "
                              f"— identifying and terminating adversary")

                        suspects = identify_adversarial_pid(
                            args.gpu, known_victim_pids={victim_pid})

                        if cuda_proc is not None and cuda_proc.poll() is None:
                            cuda_proc.terminate()
                            try:
                                cuda_proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                cuda_proc.kill()
                                cuda_proc.wait(timeout=1)
                            term_method = "SIGTERM→SIGKILL"
                        elif suspects:
                            ok, term_method = terminate_process(suspects[0]["pid"])
                        else:
                            term_method = "none"

                        mitigation_latency_ms = (time.time() - mit_start) * 1000
                        termination_done = True
                        print(f"    [+] Adversary terminated ({term_method}) in "
                              f"{mitigation_latency_ms:.1f}ms")
                        if suspects:
                            print(f"    [+] Attribution: {suspects}")
                        cuda_proc = None

                    all_records.append({
                        "trial": trial,
                        "timestamp": round(t_now - global_start, 4),
                        "phase": p_name,
                        "is_attack": int(phase["is_attack"]),
                        "category": phase["category"],
                        "data_source": hw["provenance"],
                        "power_w": round(power_w, 3),
                        "voltage_v": round(voltage_v, 5),
                        "current_a": round(current_a, 4),
                        "di_dt_amps_per_sec": round(di_dt, 4),
                        "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
                        "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                        "anomaly_detected": int(is_anom),
                        "mitigation_active": int(mitigation_active or termination_done),
                        "terminated": int(termination_done),
                        "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                        "gpu_temp_c": hw.get("gpu_temp_c", 0),
                        "gpu_util_pct": hw.get("gpu_util_pct", 0),
                        "mem_util_pct": hw.get("mem_util_pct", 0),
                        "sw_power_cap": hw.get("sw_power_cap", 0),
                        "hw_slowdown": hw.get("hw_slowdown", 0),
                        "throttle_bitmask": hw.get("throttle_bitmask", 0),
                        "adversary_pid": adversary_pid,
                        "mitigation_latency_ms": round(mitigation_latency_ms, 2) if mitigation_latency_ms else None,
                    })

                    prev_current = current_a
                    prev_time = t_now
                    time.sleep(0.02)

                # Cleanup
                if cuda_proc is not None:
                    cuda_proc.terminate()
                    try:
                        cuda_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        cuda_proc.kill()

                if mitigation_active:
                    restore_power_limit(default_limit, args.gpu)
                    mitigation_active = False

                time.sleep(3)

    finally:
        restore_power_limit(default_limit, args.gpu)
        tel_det.stop()

    # Save
    csv_path = os.path.join(out_dir, "h100_pipeline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED: {csv_path}")
    print(f"{'='*70}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Phase':>30s}  {'Power(W)':>12s}  {'Latency(ms)':>14s}  {'Throttle':>8s}")
    for phase in PHASES:
        rows = [r for r in all_records if r["phase"] == phase["name"]]
        if not rows:
            continue
        pows = [r["power_w"] for r in rows]
        lats = [r["victim_total_latency_ms"] for r in rows]
        sw = sum(r["sw_power_cap"] for r in rows)
        print(f"  {phase['name']:>30s}  "
              f"{np.mean(pows):5.1f}±{np.std(pows):4.1f}  "
              f"{np.mean(lats):7.2f}±{np.std(lats):5.2f}  "
              f"{sw:8d}")

    # Decomposition
    baseline_rows = [r for r in all_records if r["phase"] == "victim_alone"]
    attack_rows = [r for r in all_records if r["phase"] == "attack_unmitigated"]
    cap_rows = [r for r in all_records if r["phase"] == "attack_powercap_mitigated"]
    term_rows = [r for r in all_records if r["phase"] == "attack_terminated"]
    recov_rows = [r for r in all_records if r["phase"] == "recovery_baseline"]

    if baseline_rows and attack_rows:
        bl_p = np.mean([r["power_w"] for r in baseline_rows])
        at_p = np.mean([r["power_w"] for r in attack_rows])
        bl_l = np.mean([r["victim_total_latency_ms"] for r in baseline_rows])
        at_l = np.mean([r["victim_total_latency_ms"] for r in attack_rows])

        print(f"\n  Attack impact:")
        print(f"    Power:   {bl_p:.1f}W → {at_p:.1f}W  (+{(at_p-bl_p)/bl_p*100:.1f}%)")
        print(f"    Latency: {bl_l:.2f}ms → {at_l:.2f}ms  (+{(at_l-bl_l)/bl_l*100:.1f}%)")

    if cap_rows:
        cp_p = np.mean([r["power_w"] for r in cap_rows])
        cp_l = np.mean([r["victim_total_latency_ms"] for r in cap_rows])
        print(f"\n  Power-cap mitigation:")
        print(f"    Power: {cp_p:.1f}W  (reduction: {(at_p-cp_p)/at_p*100:.1f}% from attack)")
        print(f"    Latency: {cp_l:.2f}ms")

    if term_rows:
        # Post-termination samples only
        post_term = [r for r in term_rows if r["terminated"]]
        if post_term:
            tp_p = np.mean([r["power_w"] for r in post_term])
            tp_l = np.mean([r["victim_total_latency_ms"] for r in post_term])
            print(f"\n  Process termination:")
            print(f"    Post-kill power: {tp_p:.1f}W")
            print(f"    Post-kill latency: {tp_l:.2f}ms")

    if recov_rows:
        rp = np.mean([r["power_w"] for r in recov_rows])
        rl = np.mean([r["victim_total_latency_ms"] for r in recov_rows])
        print(f"\n  Recovery:")
        print(f"    Power: {rp:.1f}W  Latency: {rl:.2f}ms")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAIA H100 Closed-Loop Mitigation Pipeline")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=200.0,
                        help="Detection threshold (W)")
    parser.add_argument("--cap-ratio", type=float, default=0.6,
                        help="Power cap as fraction of default limit (e.g., 0.6 = 60%%)")
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()
    run_pipeline(args)
