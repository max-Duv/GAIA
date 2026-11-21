#!/usr/bin/env python3
"""
Project GAIA — H100 Attack Transfer Validation (Thrust T1)

Validates that GPU power-domain attack primitives transfer from the
Jetson AGX Thor (130W SoC) to discrete data-center GPUs (H100, 700W TDP).
Runs a simplified 3-condition experiment:
  1. Victim alone (baseline)
  2. Victim + cache-eviction attack
  3. Victim + combined oversubscription attack

Telemetry: pynvml (primary) or nvidia-smi (fallback).
Output: CSV compatible with the existing analysis pipeline.

Usage:
    python h100_attack_transfer.py --trials 5 --duration 15
    python h100_attack_transfer.py --victim matmul --arch sm_90
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

from realistic_victims import SyntheticMatmulVictim

_EXE = ".exe" if sys.platform == "win32" else ""


# ---------------------------------------------------------------------------
# Telemetry backend: pynvml preferred, nvidia-smi fallback
# ---------------------------------------------------------------------------

class H100Telemetry:
    def __init__(self, gpu_idx=0):
        self.gpu_idx = gpu_idx
        self._handle = None
        self._has_pynvml = False

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            self._has_pynvml = True
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"[telemetry] pynvml ready: {name}")
        except Exception as e:
            print(f"[telemetry] pynvml unavailable ({e}), falling back to nvidia-smi")

    def read(self):
        if self._has_pynvml:
            return self._read_pynvml()
        return self._read_nvidiasmi()

    def _read_pynvml(self):
        import pynvml
        h = self._handle
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
        except Exception:
            power_mw = 0
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = 0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = util.gpu
            mem_util = util.memory
        except Exception:
            gpu_util = 0
            mem_util = 0
        try:
            clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            clock = 0
        try:
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        except Exception:
            limit_mw = 700000

        throttle = 0
        try:
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        except Exception:
            pass

        sw_cap = int(bool(throttle & 0x60))
        hw_slow = int(bool(throttle & 0x18))

        return {
            "power_w": power_mw / 1000.0,
            "voltage_v": 0.0,
            "gpu_temp_c": temp,
            "gpu_util_pct": gpu_util,
            "mem_util_pct": mem_util,
            "sm_clock_mhz": clock,
            "power_limit_w": limit_mw / 1000.0,
            "sw_power_cap": sw_cap,
            "hw_slowdown": hw_slow,
            "throttle_bitmask": throttle,
            "provenance": "pynvml",
        }

    def _read_nvidiasmi(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_idx}",
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu,"
            "utilization.memory,clocks.current.sm,power.limit,"
            "clocks_throttle_reasons.sw_power_cap,"
            "clocks_throttle_reasons.hw_slowdown",
            "--format=csv,noheader,nounits",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = [x.strip() for x in r.stdout.strip().split(",")]
                return {
                    "power_w": float(parts[0]) if parts[0] not in ("[N/A]", "N/A") else 0.0,
                    "voltage_v": 0.0,
                    "gpu_temp_c": float(parts[1]) if parts[1] not in ("[N/A]", "N/A") else 0.0,
                    "gpu_util_pct": float(parts[2]) if parts[2] not in ("[N/A]", "N/A") else 0.0,
                    "mem_util_pct": float(parts[3]) if parts[3] not in ("[N/A]", "N/A") else 0.0,
                    "sm_clock_mhz": float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else 0.0,
                    "power_limit_w": float(parts[5]) if parts[5] not in ("[N/A]", "N/A") else 700.0,
                    "sw_power_cap": int(parts[6] == "Active") if len(parts) > 6 else 0,
                    "hw_slowdown": int(parts[7] == "Active") if len(parts) > 7 else 0,
                    "throttle_bitmask": 0,
                    "provenance": "nvidia-smi",
                }
        except Exception:
            pass

        return {
            "power_w": 0.0, "voltage_v": 0.0, "gpu_temp_c": 0.0,
            "gpu_util_pct": 0.0, "mem_util_pct": 0.0, "sm_clock_mhz": 0.0,
            "power_limit_w": 700.0, "sw_power_cap": 0, "hw_slowdown": 0,
            "throttle_bitmask": 0, "provenance": "none",
        }

    def get_gpu_name(self):
        if self._has_pynvml:
            import pynvml
            name = pynvml.nvmlDeviceGetName(self._handle)
            return name.decode() if isinstance(name, bytes) else name
        try:
            r = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_idx}",
                 "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return r.stdout.strip()
        except Exception:
            pass
        return "unknown"


# ---------------------------------------------------------------------------
# Platform info collection
# ---------------------------------------------------------------------------

def collect_h100_platform_info(telemetry):
    info = {
        "collection_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": sys.platform,
        "python_version": sys.version,
        "device_model": telemetry.get_gpu_name(),
    }
    for cmd_name, cmd in [
        ("nvcc_version", ["nvcc", "--version"]),
        ("nvidia_driver", ["nvidia-smi", "--query-gpu=driver_version",
                           "--format=csv,noheader"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                info[cmd_name] = r.stdout.strip()
        except Exception:
            pass
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["pytorch_cuda_version"] = torch.version.cuda or "N/A"
    except ImportError:
        pass
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5,
                           cwd=os.path.dirname(SCRIPT_DIR))
        if r.returncode == 0:
            info["git_revision"] = r.stdout.strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# CUDA compilation
# ---------------------------------------------------------------------------

def detect_gpu_arch():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            cap = r.stdout.strip().split("\n")[0].strip()
            return f"sm_{cap.replace('.', '')}"
    except Exception:
        pass
    return "sm_90"


def compile_attack_kernel(arch=None):
    src = os.path.join(T1_DIR, "gpu_power_poc_v3.cu")
    out = os.path.join(SCRIPT_DIR, "gpu_power_poc_h100" + _EXE)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Attack kernel source not found: {src}")

    if arch is None:
        arch = detect_gpu_arch()
    print(f"[compile] Compiling attack kernel for {arch}...")

    cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", out, "-lcuda"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stderr}")
    print(f"[compile] Built: {out}")
    return out


def compile_victim_kernel(arch=None):
    src = os.path.join(T1_DIR, "gpu_victim_kernel.cu")
    out = os.path.join(SCRIPT_DIR, "gpu_victim_h100" + _EXE)
    if not os.path.exists(src):
        return None

    if arch is None:
        arch = detect_gpu_arch()
    print(f"[compile] Compiling victim kernel for {arch}...")

    cmd = ["nvcc", "-O3", f"-arch={arch}", src, "-o", out, "-lcuda"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[compile] WARNING: victim kernel compile failed: {r.stderr}")
        return None
    return out


# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "name": "victim_alone",
        "is_attack": False,
        "category": "baseline",
        "mode": None,
    },
    {
        "name": "victim_cache_only",
        "is_attack": True,
        "category": "attack_cache",
        "mode": "cache",
    },
    {
        "name": "victim_compute_only",
        "is_attack": True,
        "category": "attack_compute",
        "mode": "compute",
    },
    {
        "name": "victim_sustained_stress",
        "is_attack": True,
        "category": "attack_sustained",
        "mode": "oversubscribe",
    },
    {
        "name": "victim_occupancy_stress",
        "is_attack": True,
        "category": "attack_occupancy",
        "mode": "occupancy",
    },
]


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(args):
    telemetry = H100Telemetry(gpu_idx=args.gpu)
    platform_info = collect_h100_platform_info(telemetry)

    gpu_name = platform_info.get("device_model", "unknown")
    print(f"\n{'='*70}")
    print(f"  GAIA H100 ATTACK TRANSFER VALIDATION")
    print(f"  GPU: {gpu_name}")
    print(f"  Trials: {args.trials} | Duration/phase: {args.duration}s")
    print(f"{'='*70}\n")

    attack_binary = compile_attack_kernel(args.arch)

    # H100 has 50MB L2 cache — allocate 64MB buffer to exceed it
    buf_size_mb = 64

    victim = SyntheticMatmulVictim(device_idx=args.gpu,
                                   matrix_size=args.matrix_size)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, f"results_h100_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "platform_info.json"), "w") as f:
        json.dump(platform_info, f, indent=2)

    csv_path = os.path.join(out_dir, "h100_transfer_results.csv")
    fieldnames = [
        "trial", "timestamp", "phase", "is_attack", "category",
        "data_source", "power_w", "voltage_v", "current_a",
        "di_dt_amps_per_sec", "victim_kernel_ms", "victim_total_latency_ms",
        "anomaly_detected", "mitigation_active",
        "sm_clock_mhz", "gpu_temp_c", "gpu_util_pct", "mem_util_pct",
        "sw_power_cap", "hw_slowdown", "throttle_bitmask",
    ]

    all_records = []

    for trial in range(1, args.trials + 1):
        print(f"\n{'='*60}")
        print(f"  Trial {trial}/{args.trials}")
        print(f"{'='*60}")

        for cond in CONDITIONS:
            p_name = cond["name"]
            mode = cond["mode"]
            print(f"\n  [{trial}] Phase: {p_name}", end="")
            if mode:
                print(f" (mode={mode})", end="")
            print()

            cuda_proc = None
            if mode is not None:
                stress_cmd = [
                    attack_binary,
                    "--mode", mode,
                    "--freq", "10.0",
                    "--duty", "0.5",
                    "--duration", str(int(args.duration + 5)),
                    "--streams", "8",
                    "--bufsize", str(buf_size_mb),
                ]
                cuda_proc = subprocess.Popen(
                    stress_cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True)
                print(f"    [+] Stress launched (PID {cuda_proc.pid})")
                time.sleep(1.0)

            prev_current = 0.0
            prev_time = time.time()
            phase_start = time.time()
            global_start = phase_start

            while (time.time() - phase_start) < args.duration:
                v_start = time.time()
                v_res = victim()
                t_now = time.time()
                dt = max(t_now - prev_time, 0.001)

                hw = telemetry.read()
                power_w = hw["power_w"]
                voltage_v = hw["voltage_v"]
                if voltage_v <= 0:
                    voltage_v = 12.0  # typical PCIe 12V
                current_a = power_w / voltage_v if voltage_v > 0 else 0.0
                di_dt = (current_a - prev_current) / dt

                record = {
                    "trial": trial,
                    "timestamp": round(t_now - global_start, 4),
                    "phase": p_name,
                    "is_attack": int(cond["is_attack"]),
                    "category": cond["category"],
                    "data_source": hw["provenance"],
                    "power_w": round(power_w, 3),
                    "voltage_v": round(voltage_v, 5),
                    "current_a": round(current_a, 4),
                    "di_dt_amps_per_sec": round(di_dt, 4),
                    "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
                    "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
                    "anomaly_detected": 0,
                    "mitigation_active": 0,
                    "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                    "gpu_temp_c": hw.get("gpu_temp_c", 0),
                    "gpu_util_pct": hw.get("gpu_util_pct", 0),
                    "mem_util_pct": hw.get("mem_util_pct", 0),
                    "sw_power_cap": hw.get("sw_power_cap", 0),
                    "hw_slowdown": hw.get("hw_slowdown", 0),
                    "throttle_bitmask": hw.get("throttle_bitmask", 0),
                }
                all_records.append(record)
                prev_current = current_a
                prev_time = t_now
                time.sleep(0.02)

            if cuda_proc is not None:
                cuda_proc.terminate()
                try:
                    cuda_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    cuda_proc.kill()

            # 3s cooldown between conditions
            print(f"    [~] Cooldown 3s...")
            time.sleep(3)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n{'='*70}")
    print(f"  RESULTS SAVED: {csv_path}")
    print(f"  Total samples: {len(all_records)}")
    print(f"{'='*70}")

    # Quick summary
    print(f"\n{'='*70}")
    print(f"  QUICK SUMMARY")
    print(f"{'='*70}")
    for cond in CONDITIONS:
        rows = [r for r in all_records if r["phase"] == cond["name"]]
        if not rows:
            continue
        powers = [r["power_w"] for r in rows]
        lats = [r["victim_total_latency_ms"] for r in rows]
        print(f"  {cond['name']:30s}  "
              f"power={np.mean(powers):6.1f}±{np.std(powers):5.1f} W  "
              f"latency={np.mean(lats):8.2f}±{np.std(lats):6.2f} ms")

    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA H100 Attack Transfer Validation")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--victim", type=str, default="matmul",
                        choices=["matmul"])
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()
    run_experiment(args)
