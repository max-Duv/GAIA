"""
Project GAIA: Blinded Multi-Process Adversary Attribution Experiment
====================================================================
Addresses proposal commitment P15 and cross-examiner finding #5:
the closed-loop mitigator must identify the adversarial process
WITHOUT using ground-truth PIDs.

Design:
  - Multiple legitimate GPU processes run concurrently (training,
    inference, and optionally a high-power workload).
  - An attacker is injected after a baseline period.
  - A blinded attribution system enumerates GPU processes, measures
    per-process power contribution via sequential suspension, and
    terminates the highest-contribution candidate.
  - Metrics: attribution accuracy, innocent kills, attribution
    latency, power recovery, victim QoS.

Usage:
    sudo python blinded_attribution_experiment.py [--trials 5] [--duration 20]
"""

import os
import sys
import time
import json
import signal
import subprocess
import csv
import argparse
import threading
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "T1"))
sys.path.insert(0, os.path.join(PARENT_DIR, "T2"))

_EXE = ".exe" if sys.platform == "win32" else ""

from gpu_fault_assessor_v2 import GPUFaultAssessorV2, CONDITION_LABELS, collect_platform_info
from gpu_defense_filters_v2 import PlatformAgnosticAnomalyDetector

from realistic_victims import TrainingVictim, InferenceVictim, LegitimateHighPowerWorkload, _ensure_torch
_PYTORCH_VICTIMS = _ensure_torch()


BLINDED_CONDITION_LABELS = {
    "multi_baseline":                {"is_attack": False, "category": "blinded_baseline"},
    "multi_attack_undetected":       {"is_attack": True,  "category": "blinded_attack"},
    "multi_attack_blinded_response": {"is_attack": True,  "category": "blinded_response"},
    "multi_recovery":                {"is_attack": False, "category": "blinded_recovery"},
}
CONDITION_LABELS.update(BLINDED_CONDITION_LABELS)


def enumerate_gpu_processes(exclude_pids=None):
    """Discover PIDs currently using the GPU, without knowing which is adversarial.

    Tries multiple methods in order of reliability on Jetson Thor:
      1. nvidia-smi pmon (works on desktop and some Jetson firmwares)
      2. /proc scan for /dev/nvidia* file descriptors
      3. lsof /dev/nvidia* fallback

    Returns list of dicts: [{"pid": int, "name": str, "method": str}, ...]
    """
    if exclude_pids is None:
        exclude_pids = set()
    else:
        exclude_pids = set(exclude_pids)

    our_pid = os.getpid()
    exclude_pids.add(our_pid)

    discovered = {}

    # Method 1: nvidia-smi pmon
    try:
        res = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        if res.returncode == 0:
            for line in res.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        if pid > 0 and pid not in exclude_pids:
                            name = parts[-1] if len(parts) >= 8 else "unknown"
                            discovered[pid] = {"pid": pid, "name": name, "method": "pmon"}
                    except ValueError:
                        continue
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: /proc scan for nvidia device fds
    if not discovered:
        try:
            for entry in os.listdir("/proc"):
                try:
                    pid = int(entry)
                except ValueError:
                    continue
                if pid in exclude_pids:
                    continue
                fd_dir = f"/proc/{pid}/fd"
                try:
                    for fd in os.listdir(fd_dir):
                        try:
                            target = os.readlink(os.path.join(fd_dir, fd))
                            if "/dev/nvidia" in target or "/dev/dri/renderD" in target:
                                try:
                                    with open(f"/proc/{pid}/comm") as f:
                                        name = f.read().strip()
                                except (IOError, OSError):
                                    name = "unknown"
                                discovered[pid] = {"pid": pid, "name": name, "method": "proc_scan"}
                                break
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    continue
        except OSError:
            pass

    # Method 3: lsof fallback
    if not discovered:
        try:
            res = subprocess.run(
                ["lsof", "+D", "/dev/nvidia0"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            if pid not in exclude_pids:
                                discovered[pid] = {
                                    "pid": pid,
                                    "name": parts[0],
                                    "method": "lsof",
                                }
                        except ValueError:
                            continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return list(discovered.values())


def measure_power_mean(assessor, duration=2.0, interval=0.05):
    """Collect power readings and return mean."""
    readings = []
    start = time.time()
    while time.time() - start < duration:
        hw = assessor.get_hardware_telemetry()
        readings.append(hw["power_w"])
        time.sleep(interval)
    return float(np.mean(readings)) if readings else 0.0


def blinded_attribute_and_kill(assessor, known_legit_pids, ground_truth_attacker_pid,
                               baseline_power_w=None, measure_duration=2.0):
    """Blinded adversary attribution via sequential elimination.

    On Jetson Thor the attacker's duty-cycled GPU kernel causes DVFS
    clock contention that REDUCES total power.  Killing the attacker
    therefore INCREASES power back toward the baseline.  We exploit this:
    kill each candidate (newest PID first), measure power, and attribute
    to the process whose removal moves power closest to baseline.

    If baseline_power_w is not provided, falls back to largest absolute
    power change.
    """
    attribution_start = time.time()

    SYSTEM_PROCESS_NAMES = {"Xorg", "gnome-shell", "gdm", "Xwayland",
                            "kwin_wayland", "sddm", "lightdm", "compiz"}

    candidates = enumerate_gpu_processes(exclude_pids=known_legit_pids)
    enum_time = time.time() - attribution_start

    if not candidates:
        return {
            "success": False,
            "reason": "no_candidates_found",
            "attribution_latency_s": time.time() - attribution_start,
            "candidates_found": 0,
            "innocent_kills": 0,
            "correct_attribution": False,
            "killed_pid": None,
            "enum_method": "none",
        }

    print(f"    [attrib] Found {len(candidates)} candidate GPU process(es) "
          f"(enum took {enum_time:.3f}s)")
    for c in candidates:
        print(f"      PID {c['pid']:>7d}  {c['name']:<24s}  (via {c['method']})")

    non_system = [c for c in candidates if c["name"] not in SYSTEM_PROCESS_NAMES]
    skipped = [c for c in candidates if c["name"] in SYSTEM_PROCESS_NAMES]
    if skipped:
        print(f"    [attrib] Skipping {len(skipped)} system process(es): "
              + ", ".join(f"{c['name']}({c['pid']})" for c in skipped))

    if not non_system:
        return {
            "success": False,
            "reason": "only_system_processes",
            "attribution_latency_s": time.time() - attribution_start,
            "candidates_found": len(candidates),
            "innocent_kills": 0,
            "correct_attribution": False,
            "killed_pid": None,
            "enum_method": candidates[0]["method"] if candidates else "none",
        }

    non_system.sort(key=lambda c: c["pid"], reverse=True)

    if len(non_system) == 1:
        suspect = non_system[0]
        innocent_count = 0
        print(f"    [attrib] Single non-system candidate PID {suspect['pid']} "
              f"— terminating directly")
        try:
            os.kill(suspect["pid"], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1.0)
    else:
        pre_power = measure_power_mean(assessor, duration=measure_duration)
        anomaly_dist = abs(pre_power - baseline_power_w) if baseline_power_w else 0
        print(f"    [attrib] Pre-attribution power: {pre_power:.2f} W"
              + (f"  (baseline: {baseline_power_w:.1f} W, "
                 f"deviation: {anomaly_dist:.1f} W)"
                 if baseline_power_w else ""))
        print(f"    [attrib] Sequential elimination ({len(non_system)} candidates, "
              f"newest PID first)...")

        kill_results = []
        prev_power = pre_power

        for i, cand in enumerate(non_system):
            pid = cand["pid"]
            print(f"      [{i+1}/{len(non_system)}] Killing PID {pid} "
                  f"({cand['name']})...", end="", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                print(f" failed ({e})")
                kill_results.append((cand, prev_power, prev_power, 0.0))
                continue

            time.sleep(2.0)
            post_power = measure_power_mean(assessor, duration=measure_duration)
            delta = post_power - prev_power
            print(f" power={post_power:.2f}W (Δ{delta:+.2f}W)")

            kill_results.append((cand, prev_power, post_power, delta))
            prev_power = post_power

        if baseline_power_w is not None:
            best_idx = -1
            best_recovery = -1.0
            running_power = pre_power
            for idx, (cand, pre, post, delta) in enumerate(kill_results):
                pre_dist = abs(running_power - baseline_power_w)
                post_dist = abs(post - baseline_power_w)
                recovery = pre_dist - post_dist
                running_power = post
                print(f"      PID {cand['pid']}: deviation {pre_dist:.1f}W → "
                      f"{post_dist:.1f}W (recovery={recovery:+.1f}W)")
                if recovery > best_recovery:
                    best_recovery = recovery
                    best_idx = idx
        else:
            best_idx = int(np.argmax([abs(kr[3]) for kr in kill_results]))

        suspect = kill_results[best_idx][0]
        innocent_count = len(kill_results) - 1
        print(f"    [attrib] Best recovery from PID {suspect['pid']} "
              f"— identified as adversary")

    killed_pid = suspect["pid"]
    attribution_latency = time.time() - attribution_start
    correct = (killed_pid == ground_truth_attacker_pid)

    print(f"    [attrib] Attributed PID {killed_pid} "
          f"({innocent_count} innocent kill(s))")
    print(f"    [attrib] Correct attribution: {correct} "
          f"(ground truth: {ground_truth_attacker_pid})")
    print(f"    [attrib] Total attribution latency: {attribution_latency:.3f}s")

    return {
        "success": True,
        "reason": "attributed",
        "attribution_latency_s": round(attribution_latency, 4),
        "candidates_found": len(candidates),
        "innocent_kills": innocent_count,
        "correct_attribution": correct,
        "killed_pid": killed_pid,
        "ground_truth_pid": ground_truth_attacker_pid,
        "kill_method": "SIGKILL",
        "kill_elapsed_s": round(attribution_latency, 4),
        "enum_method": candidates[0]["method"] if candidates else "none",
        "baseline_power_w": baseline_power_w,
    }


class VictimRunner:
    """Runs a victim workload continuously in a background thread, collecting latencies."""

    def __init__(self, name, victim_callable):
        self.name = name
        self.victim = victim_callable
        self.latencies = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while self._running:
            try:
                result = self.victim()
                self.latencies.append({
                    "timestamp": time.time(),
                    "total_latency_ms": result["total_latency_ms"],
                    "gpu_compute_ms": result["gpu_compute_ms"],
                    "backend": result.get("victim_backend", self.name),
                })
            except Exception:
                time.sleep(0.1)

    def get_recent_latency(self, last_n=10):
        if not self.latencies:
            return 0.0
        recent = self.latencies[-last_n:]
        return float(np.mean([r["total_latency_ms"] for r in recent]))

    def clear_latencies(self):
        self.latencies.clear()

    @property
    def pid(self):
        return os.getpid()


class SubprocessVictimRunner:
    """Runs the compiled CUDA victim binary as a separate process with its own PID.

    Used when PyTorch is not available. Each instance runs the victim kernel
    in a loop, giving the attribution system distinct PIDs to sort through.
    """

    def __init__(self, name, victim_binary, elements=16777216, inner_iters=64):
        self.name = name
        self._binary = victim_binary
        self._elements = elements
        self._inner_iters = inner_iters
        self._proc = None
        self.latencies = []
        self._running = False
        self._monitor_thread = None

    def start(self):
        from gpu_fault_assessor_v2 import _cuda_subprocess_env
        cmd = [
            self._binary,
            "--reps", "999999",
            "--elements", str(self._elements),
            "--inner-iters", str(self._inner_iters),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=_cuda_subprocess_env(),
        )
        self._running = True
        self._monitor_thread = threading.Thread(target=self._read_output, daemon=True)
        self._monitor_thread.start()
        print(f"    [victim-proc] '{self.name}' started as PID {self._proc.pid}")

    def _read_output(self):
        import re
        pattern = re.compile(r"VICTIM_LATENCY_MS:\s*([\d.]+)")
        while self._running and self._proc and self._proc.poll() is None:
            try:
                line = self._proc.stdout.readline()
                if not line:
                    break
                m = pattern.search(line)
                if m:
                    ms = float(m.group(1))
                    self.latencies.append({
                        "timestamp": time.time(),
                        "total_latency_ms": ms,
                        "gpu_compute_ms": ms,
                        "backend": f"cuda_subprocess_{self.name}",
                    })
            except Exception:
                break

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def get_recent_latency(self, last_n=10):
        if not self.latencies:
            return 0.0
        recent = self.latencies[-last_n:]
        return float(np.mean([r["total_latency_ms"] for r in recent]))

    def clear_latencies(self):
        self.latencies.clear()

    @property
    def pid(self):
        return self._proc.pid if self._proc else 0


def run_blinded_attribution_experiment(
    cuda_binary=None,
    duration_per_phase=20.0,
    detection_threshold_w=60.0,
    output_csv=None,
    gpu_idx=0,
    num_trials=5,
    include_highpower=False,
):
    if cuda_binary is None:
        cuda_binary = os.path.join(PARENT_DIR, "T1", "gpu_power_poc_v3" + _EXE)
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "blinded_attribution.csv")

    if not os.path.exists(cuda_binary):
        raise FileNotFoundError(
            f"CUDA stress binary not found: {cuda_binary}\n"
            f"Run T1 first to compile it."
        )

    print("=====================================================================")
    print(" Project GAIA: Blinded Multi-Process Attribution Experiment          ")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s")
    print(f" High-power legitimate workload: {'YES' if include_highpower else 'NO'}")
    print("=====================================================================")

    assessor = GPUFaultAssessorV2(target_gpu_index=gpu_idx)

    print("[+] Initializing legitimate workloads...")
    victim_binary = os.path.join(PARENT_DIR, "T1", "gpu_victim_kernel" + _EXE)
    use_subprocess_victims = not _PYTORCH_VICTIMS

    if use_subprocess_victims:
        if not os.path.exists(victim_binary):
            from realistic_victims import SyntheticMatmulVictim
            _tmp = SyntheticMatmulVictim(device_idx=gpu_idx)
            del _tmp
        if not os.path.exists(victim_binary):
            raise FileNotFoundError(
                f"Victim binary not found: {victim_binary}\n"
                "Compile gpu_victim_kernel.cu first.")
        print("    [!] PyTorch not available — using subprocess CUDA victims (separate PIDs)")

    if use_subprocess_victims:
        training_victim = None
        inference_victim = None
        highpower_victim = None
    else:
        training_victim = TrainingVictim(device_idx=gpu_idx)
        inference_victim = InferenceVictim(device_idx=gpu_idx)
        highpower_victim = LegitimateHighPowerWorkload(device_idx=gpu_idx) if include_highpower else None

    output_dir = os.path.dirname(output_csv) or "."
    platform_info = collect_platform_info()
    platform_info["experiment"] = "blinded_attribution"
    platform_info["num_trials"] = num_trials
    platform_info["duration_per_phase"] = duration_per_phase
    platform_info["detection_threshold_w"] = detection_threshold_w
    platform_info["include_highpower"] = include_highpower
    pinfo_path = os.path.join(output_dir, "blinded_attribution_platform_info.json")
    with open(pinfo_path, "w") as f:
        json.dump(platform_info, f, indent=2, default=str)
    print(f"[+] Platform info saved to {pinfo_path}")

    stress_cmd = [
        cuda_binary,
        "--mode", "oversubscribe",
        "--freq", "10.0",
        "--duty", "0.5",
        "--duration", str(int(duration_per_phase * 3 + 30)),
    ]

    all_records = []
    trial_summaries = []
    attribution_log = []

    our_pid = os.getpid()

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'='*69}")
        print(f"  Trial {trial_idx}/{num_trials}")
        print(f"{'='*69}")

        detector = PlatformAgnosticAnomalyDetector(
            sampling_rate_hz=4.0,
            power_anomaly_threshold_w=detection_threshold_w,
            external_telemetry=True,
        )

        if use_subprocess_victims:
            runners = [
                SubprocessVictimRunner("training", victim_binary,
                                       elements=16777216, inner_iters=64),
                SubprocessVictimRunner("inference", victim_binary,
                                       elements=4194304, inner_iters=16),
            ]
            if include_highpower:
                runners.append(SubprocessVictimRunner(
                    "highpower", victim_binary,
                    elements=67108864, inner_iters=128))
        else:
            runners = [
                VictimRunner("training", training_victim),
                VictimRunner("inference", inference_victim),
            ]
            if highpower_victim:
                runners.append(VictimRunner("highpower", highpower_victim))

        known_legit_pids = {our_pid}

        for r in runners:
            r.start()
        time.sleep(1.0)

        global_start = time.time()
        prev_current_a = 0.0
        prev_time = global_start
        cuda_proc = None
        baseline_power_w = None
        attacker_pid = None
        attribution_result = None

        phases = [
            {"name": "multi_baseline",
             "duration": duration_per_phase, "launch_attacker": False,
             "do_attribution": False},
            {"name": "multi_attack_undetected",
             "duration": duration_per_phase, "launch_attacker": True,
             "do_attribution": False},
            {"name": "multi_attack_blinded_response",
             "duration": duration_per_phase * 1.5, "launch_attacker": True,
             "do_attribution": True},
            {"name": "multi_recovery",
             "duration": duration_per_phase, "launch_attacker": False,
             "do_attribution": False},
        ]

        for p_info in phases:
            p_name = p_info["name"]
            p_duration = p_info["duration"]
            print(f"\n  [+] Phase: '{p_name}' ({p_duration:.0f}s)")

            for r in runners:
                r.clear_latencies()

            detector.power_buffer.clear()
            detector.voltage_buffer.clear()
            detector.time_buffer.clear()

            label_info = BLINDED_CONDITION_LABELS.get(
                p_name, {"is_attack": False, "category": "unknown"})
            is_attack = int(label_info["is_attack"])

            if p_info["launch_attacker"] and cuda_proc is None:
                from gpu_fault_assessor_v2 import _cuda_subprocess_env
                cuda_proc = subprocess.Popen(
                    stress_cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                    env=_cuda_subprocess_env(),
                )
                attacker_pid = cuda_proc.pid
                print(f"    [+] Attacker launched (PID: {attacker_pid}) "
                      f"— NOT shared with attribution system")
                time.sleep(0.5)

            if not p_info["launch_attacker"] and cuda_proc is not None:
                if cuda_proc.poll() is None:
                    cuda_proc.terminate()
                    try:
                        cuda_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        cuda_proc.kill()
                cuda_proc = None
                attacker_pid = None

            did_attribution = False
            phase_start = time.time()

            while (time.time() - phase_start) < p_duration:
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
                is_anomalous, mean_power_det, _ = detector.analyze_power_level()

                avg_lat = np.mean([r.get_recent_latency() for r in runners])

                all_records.append({
                    "trial": trial_idx,
                    "timestamp": round(t_curr - global_start, 4),
                    "phase": p_name,
                    "is_attack": is_attack,
                    "category": label_info["category"],
                    "data_source": hw["provenance"],
                    "power_w": round(p_w, 3),
                    "voltage_v": round(v_meas, 5),
                    "current_a": round(curr_a, 4),
                    "di_dt_amps_per_sec": round(di_dt, 4),
                    "avg_victim_latency_ms": round(avg_lat, 3),
                    "anomaly_detected": int(is_anomalous),
                    "attribution_done": int(did_attribution),
                    "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                    "gpu_temp_c": hw.get("gpu_temp_c", 0),
                    "gpu_util_pct": hw.get("gpu_util_pct"),
                    "mem_util_pct": hw.get("mem_util_pct"),
                    "sw_power_cap": hw.get("sw_power_cap", 0),
                    "hw_slowdown": hw.get("hw_slowdown", 0),
                    "num_legit_processes": len(runners),
                    "attacker_pid": attacker_pid,
                })

                if (p_info["do_attribution"] and is_anomalous
                        and not did_attribution and attacker_pid is not None):
                    print(f"    [!] Anomaly detected (mean: {mean_power_det:.1f} W) "
                          f"— launching BLINDED attribution")
                    attribution_result = blinded_attribute_and_kill(
                        assessor=assessor,
                        known_legit_pids=known_legit_pids,
                        ground_truth_attacker_pid=attacker_pid,
                        baseline_power_w=baseline_power_w,
                    )
                    attribution_result["trial"] = trial_idx
                    attribution_result["phase"] = p_name
                    attribution_result["detection_power_w"] = round(mean_power_det, 2)
                    attribution_log.append(attribution_result)
                    did_attribution = True

                    if attribution_result.get("correct_attribution"):
                        cuda_proc = None

                prev_current_a = curr_a
                prev_time = t_curr
                time.sleep(0.05)

            phase_records = [r for r in all_records
                            if r["trial"] == trial_idx and r["phase"] == p_name]
            if phase_records:
                mean_pow = np.mean([r["power_w"] for r in phase_records])
                mean_lat = np.mean([r["avg_victim_latency_ms"] for r in phase_records])
                print(f"    Phase '{p_name}': power={mean_pow:.2f}W, "
                      f"avg_victim_lat={mean_lat:.2f}ms")
                if p_name == "multi_baseline":
                    baseline_power_w = float(mean_pow)

        for r in runners:
            r.stop()

        if cuda_proc is not None and cuda_proc.poll() is None:
            cuda_proc.terminate()
            try:
                cuda_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cuda_proc.kill()
            cuda_proc = None

        trial_summary = {"trial": trial_idx}
        for p_name in ["multi_baseline", "multi_attack_undetected",
                        "multi_attack_blinded_response", "multi_recovery"]:
            phase_recs = [r for r in all_records
                          if r["trial"] == trial_idx and r["phase"] == p_name]
            if phase_recs:
                trial_summary[f"{p_name}_power_w"] = round(
                    np.mean([r["power_w"] for r in phase_recs]), 3)
                trial_summary[f"{p_name}_latency_ms"] = round(
                    np.mean([r["avg_victim_latency_ms"] for r in phase_recs]), 3)

        if attribution_result:
            trial_summary["correct_attribution"] = attribution_result["correct_attribution"]
            trial_summary["innocent_kills"] = attribution_result["innocent_kills"]
            trial_summary["attribution_latency_s"] = attribution_result["attribution_latency_s"]
            trial_summary["candidates_found"] = attribution_result["candidates_found"]
        trial_summaries.append(trial_summary)

    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[+] Telemetry saved to: {output_csv} ({len(all_records)} rows)")

    attrib_path = output_csv.replace(".csv", "_attribution_log.json")
    with open(attrib_path, "w") as f:
        json.dump(attribution_log, f, indent=2, default=str)
    print(f"[+] Attribution log saved to: {attrib_path}")

    summary_csv = output_csv.replace(".csv", "_summary.csv")
    if trial_summaries:
        fieldnames = list(trial_summaries[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trial_summaries)
        print(f"[+] Trial summary saved to: {summary_csv}")

    _print_aggregate_results(trial_summaries, attribution_log, num_trials)

    return trial_summaries, attribution_log


def _print_aggregate_results(trial_summaries, attribution_log, num_trials):
    print("\n=====================================================================")
    print("  PROJECT GAIA: BLINDED ATTRIBUTION EXPERIMENT SUMMARY              ")
    print(f"  Aggregated over {num_trials} trial(s)")
    print("=====================================================================")

    phase_names = ["multi_baseline", "multi_attack_undetected",
                   "multi_attack_blinded_response", "multi_recovery"]

    print(f"\n  {'Phase':<36}{'Power (W)':<18}{'Victim Lat (ms)':<18}")
    print("  " + "-" * 70)
    for pn in phase_names:
        pows = [s.get(f"{pn}_power_w", 0) for s in trial_summaries
                if f"{pn}_power_w" in s]
        lats = [s.get(f"{pn}_latency_ms", 0) for s in trial_summaries
                if f"{pn}_latency_ms" in s]
        if pows:
            pow_str = f"{np.mean(pows):.2f} +/- {np.std(pows, ddof=1):.2f}" if len(pows) > 1 else f"{np.mean(pows):.2f}"
            lat_str = f"{np.mean(lats):.2f} +/- {np.std(lats, ddof=1):.2f}" if len(lats) > 1 else f"{np.mean(lats):.2f}"
            print(f"  {pn:<36}{pow_str:<18}{lat_str:<18}")

    if attribution_log:
        correct = sum(1 for a in attribution_log if a.get("correct_attribution"))
        total = len(attribution_log)
        innocent = sum(a.get("innocent_kills", 0) for a in attribution_log)
        latencies = [a["attribution_latency_s"] for a in attribution_log
                     if a.get("success")]
        candidates = [a["candidates_found"] for a in attribution_log
                      if a.get("success")]

        print(f"\n  Attribution Results:")
        print(f"    Correct attributions:   {correct}/{total} "
              f"({correct/total*100:.0f}%)" if total else "    No attributions")
        print(f"    Innocent process kills: {innocent}")
        if latencies:
            print(f"    Attribution latency:    {np.mean(latencies):.2f} +/- "
                  f"{np.std(latencies, ddof=1):.2f} s")
        if candidates:
            print(f"    Candidates per trial:   {np.mean(candidates):.1f} +/- "
                  f"{np.std(candidates, ddof=1):.1f}")

        if any(not a.get("correct_attribution") for a in attribution_log):
            print(f"\n  WARNING: Incorrect attributions detected!")
            for a in attribution_log:
                if not a.get("correct_attribution"):
                    print(f"    Trial {a.get('trial')}: killed PID {a.get('killed_pid')}, "
                          f"actual attacker PID {a.get('ground_truth_pid')}")

    # Power recovery analysis
    baseline_pows = [s.get("multi_baseline_power_w", 0)
                     for s in trial_summaries if "multi_baseline_power_w" in s]
    attack_pows = [s.get("multi_attack_undetected_power_w", 0)
                   for s in trial_summaries if "multi_attack_undetected_power_w" in s]
    recovery_pows = [s.get("multi_recovery_power_w", 0)
                     for s in trial_summaries if "multi_recovery_power_w" in s]

    if baseline_pows and attack_pows and recovery_pows:
        print(f"\n  Power Recovery:")
        print(f"    Baseline:  {np.mean(baseline_pows):.2f} W")
        print(f"    Attack:    {np.mean(attack_pows):.2f} W")
        print(f"    Recovery:  {np.mean(recovery_pows):.2f} W")
        recovery_pct = ((np.mean(attack_pows) - np.mean(recovery_pows))
                        / max(np.mean(attack_pows) - np.mean(baseline_pows), 0.01) * 100)
        print(f"    Recovery:  {recovery_pct:.1f}% of attack-induced elevation removed")

    print("\n=====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project GAIA: Blinded Multi-Process Attribution Experiment")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Duration per phase in seconds (default: 20)")
    parser.add_argument("--det-threshold", type=float, default=60.0,
                        help="Power anomaly detection threshold in watts")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of repeated trials (default: 5)")
    parser.add_argument("--include-highpower", action="store_true",
                        help="Include a legitimate high-power workload (hard case)")
    args = parser.parse_args()

    run_blinded_attribution_experiment(
        duration_per_phase=args.duration,
        detection_threshold_w=args.det_threshold,
        output_csv=args.out,
        gpu_idx=args.gpu,
        num_trials=args.trials,
        include_highpower=args.include_highpower,
    )
