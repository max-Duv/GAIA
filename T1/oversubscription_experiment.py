"""
Project GAIA: Oversubscription & Power-Budget Stress Experiment (P07)
=====================================================================
Addresses proposal commitment P07: "Study oversubscription and instability risk."

Strategy: Constrain the platform's effective power budget below the combined
attacker+victim demand using multiple mechanisms available on Jetson AGX Thor:

  1. nvpmodel power modes (whole-module power/clock envelope)
  2. devfreq max_freq capping (GPU clock ceiling -> indirect power ceiling)

At each budget level, runs victim-only and victim+attacker conditions,
measuring whether throttling events (sw_power_cap, hw_slowdown), voltage
sag, or latency degradation occur — the oversubscription evidence the
cross-examiner found missing.

Usage (on Jetson via SSH):
    sudo python3 oversubscription_experiment.py --trials 5 --duration 15
    sudo python3 oversubscription_experiment.py --trials 5 --budget 90,70,50
    sudo python3 oversubscription_experiment.py --trials 5 --strategy nvpmodel
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

from gpu_fault_assessor_v2 import (
    GPUFaultAssessorV2, CONDITION_LABELS, collect_platform_info,
    _cuda_subprocess_env,
)
from controlled_3phase_runner_v2 import (
    compile_cuda_backend, detect_gpu_arch,
    _read_devfreq, _write_devfreq, devfreq_snapshot, devfreq_restore,
    devfreq_pick_fixed_freq,
)

_EXE = ".exe" if sys.platform == "win32" else ""


# ---------------------------------------------------------------------------
# nvpmodel helpers
# ---------------------------------------------------------------------------

def query_nvpmodel():
    """Return current nvpmodel mode id and name, or None if unavailable."""
    try:
        r = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        lines = r.stdout.strip().splitlines()
        mode_id = None
        mode_name = None
        for line in lines:
            low = line.lower()
            if "power mode" in low or "nvpmodel" in low:
                for tok in line.split():
                    if tok.isdigit():
                        mode_id = int(tok)
                        break
            if "mode:" in low:
                mode_name = line.split(":", 1)[-1].strip()
        return {"mode_id": mode_id, "mode_name": mode_name, "raw": r.stdout.strip()}
    except Exception:
        return None


def list_nvpmodel_modes():
    """List available nvpmodel power modes with their power caps.
    Returns list of dicts with mode_id and description."""
    try:
        r = subprocess.run(
            ["nvpmodel", "-p"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        modes = []
        for line in r.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            for i, p in enumerate(parts):
                if p.isdigit():
                    modes.append({
                        "mode_id": int(p),
                        "description": line,
                    })
                    break
        return modes
    except Exception:
        return []


def set_nvpmodel(mode_id):
    """Switch nvpmodel power mode. Requires sudo."""
    cmd = ["sudo", "nvpmodel", "-m", str(mode_id)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        print(f"    [!] nvpmodel -m {mode_id} failed: {r.stderr.strip()}")
        return False
    time.sleep(2)
    print(f"    [+] nvpmodel set to mode {mode_id}")
    return True


# ---------------------------------------------------------------------------
# devfreq-based power budget approximation
# ---------------------------------------------------------------------------

def get_available_freqs(devfreq_path):
    """Return sorted list of available GPU frequencies in Hz."""
    try:
        avail = _read_devfreq(devfreq_path, 'available_frequencies')
        return sorted(int(f) for f in avail.split())
    except (IOError, OSError, ValueError):
        return []


def set_devfreq_max(devfreq_path, freq_hz):
    """Cap the GPU max clock. Uses sudo sh -c for sysfs write."""
    freq_path = os.path.join(devfreq_path, 'max_freq')
    try:
        with open(freq_path, 'w') as f:
            f.write(str(int(freq_hz)))
        actual = int(_read_devfreq(devfreq_path, 'cur_freq'))
        print(f"    [+] GPU max_freq capped to {freq_hz/1e6:.0f} MHz (cur: {actual/1e6:.0f} MHz)")
        return True
    except (IOError, OSError) as e:
        cmd = ['sudo', 'sh', '-c', f'echo {int(freq_hz)} > {freq_path}']
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            print(f"    [!] Failed to set max_freq to {freq_hz/1e6:.0f} MHz: {r.stderr.strip()}")
            return False
        print(f"    [+] GPU max_freq capped to {freq_hz/1e6:.0f} MHz (via sudo)")
        return True


def compute_freq_budget_levels(available_freqs, num_levels=5):
    """Select frequency levels spanning max down to ~30% of max.
    These serve as indirect power budget proxies since power ~ f * V^2."""
    if not available_freqs:
        return []
    max_f = available_freqs[-1]
    targets = []
    for pct in [100, 80, 60, 40, 30]:
        target = int(max_f * pct / 100)
        closest = min(available_freqs, key=lambda f: abs(f - target))
        if closest not in targets:
            targets.append(closest)
    return sorted(targets, reverse=True)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_oversubscription_trial(
    assessor, cuda_binary, victim_fn, devfreq_path,
    freq_cap_hz, budget_label, duration_per_phase, trial_num,
    run_attacker=True,
):
    """Run one trial at a given power budget level.
    Returns (records, phase_summary)."""

    records = []
    trial_start = time.time()
    prev_current_a = 0.0
    prev_time = trial_start

    phase_name = f"budget_{budget_label}_{'attack' if run_attacker else 'baseline'}"
    is_attack = 1 if run_attacker else 0
    category = "oversub_attack" if run_attacker else "oversub_baseline"

    print(f"  [T{trial_num}] Phase '{phase_name}' (cap: {freq_cap_hz/1e6:.0f} MHz, "
          f"attacker: {run_attacker})...")

    cuda_proc = None
    stress_start_ts = None

    if run_attacker:
        cmd = [
            cuda_binary,
            "--mode", "oversubscribe",
            "--freq", "0.001",
            "--duty", "1.0",
            "--duration", str(duration_per_phase + 3.0),
        ]
        try:
            cuda_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=_cuda_subprocess_env(),
            )
            stress_start_ts = time.time()
            time.sleep(0.5)
        except Exception as e:
            print(f"    [!] Failed to spawn stress process: {e}")

    latencies_tot = []
    latencies_compute = []
    phase_start = time.time()

    while (time.time() - phase_start) < duration_per_phase:
        v_res = victim_fn()
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

        devfreq_cur = None
        if devfreq_path:
            try:
                devfreq_cur = int(_read_devfreq(devfreq_path, 'cur_freq'))
            except (IOError, OSError, ValueError):
                pass

        records.append({
            "trial": trial_num,
            "timestamp": round(t_curr - trial_start, 4),
            "phase": phase_name,
            "is_attack": is_attack,
            "category": category,
            "budget_label": budget_label,
            "freq_cap_hz": freq_cap_hz,
            "freq_cap_mhz": round(freq_cap_hz / 1e6, 0),
            "data_source": hw["provenance"],
            "power_w": round(p_w, 3),
            "voltage_v": round(v_meas, 5),
            "current_a": round(curr_a, 4),
            "di_dt_amps_per_sec": round(di_dt, 4),
            "victim_kernel_ms": round(v_res["gpu_compute_ms"], 3),
            "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
            "victim_launch_delay_ms": round(v_res.get("launch_delay_ms", 0), 3),
            "victim_backend": v_res.get("victim_backend", "unknown"),
            "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
            "gpu_temp_c": hw.get("gpu_temp_c", 0),
            "gpu_util_pct": hw.get("gpu_util_pct"),
            "mem_util_pct": hw.get("mem_util_pct"),
            "sw_power_cap": hw.get("sw_power_cap", 0),
            "hw_slowdown": hw.get("hw_slowdown", 0),
            "throttle_bitmask": hw.get("throttle_bitmask", 0),
            "devfreq_cur_hz": devfreq_cur,
            "stress_start_ts": round(stress_start_ts - trial_start, 4) if stress_start_ts else None,
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

    phase_records = [r for r in records if r["phase"] == phase_name]
    cap_count = sum(
        1 for r in phase_records
        if r["sw_power_cap"] == 1 or r["hw_slowdown"] == 1
    )

    summary = {
        "phase": phase_name,
        "budget_label": budget_label,
        "freq_cap_mhz": round(freq_cap_hz / 1e6, 0),
        "is_attack": is_attack,
        "mean_total_latency_ms": float(np.mean(latencies_tot)),
        "std_total_latency_ms": float(np.std(latencies_tot, ddof=1)) if len(latencies_tot) > 1 else 0.0,
        "mean_gpu_compute_ms": float(np.mean(latencies_compute)),
        "mean_power_w": float(np.mean([r["power_w"] for r in phase_records])),
        "std_power_w": float(np.std([r["power_w"] for r in phase_records], ddof=1)) if len(phase_records) > 1 else 0.0,
        "max_power_w": float(np.max([r["power_w"] for r in phase_records])),
        "min_voltage_v": float(np.min([r["voltage_v"] for r in phase_records])),
        "capping_rate_pct": (cap_count / max(len(phase_records), 1)) * 100.0,
        "mean_clock_mhz": float(np.mean([r["sm_clock_mhz"] for r in phase_records if r["sm_clock_mhz"] > 0])) if any(r["sm_clock_mhz"] > 0 for r in phase_records) else 0.0,
        "n_samples": len(phase_records),
    }

    print(f"    Latency: {summary['mean_total_latency_ms']:.2f} ± {summary['std_total_latency_ms']:.2f} ms | "
          f"Power: {summary['mean_power_w']:.2f} W (max {summary['max_power_w']:.1f}) | "
          f"Cap: {summary['capping_rate_pct']:.1f}% | "
          f"Clock: {summary['mean_clock_mhz']:.0f} MHz")

    return records, summary


def run_oversubscription_experiment(
    cuda_binary=None,
    duration_per_phase=15.0,
    output_csv=None,
    num_trials=5,
    arch_override=None,
    seed=None,
    victim_type="synthetic",
    budget_freqs_mhz=None,
    strategy="devfreq",
    cooldown_between=5.0,
):
    if cuda_binary is None:
        cuda_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3" + _EXE)
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "..", "..", "results", "latest",
                                  "t1_oversubscription.csv")

    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = output_csv.replace(".csv", "_summary.csv")

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    print("=" * 69)
    print(" Project GAIA: Oversubscription Experiment (P07)")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s | Seed: {seed}")
    print(f" Victim: {victim_type} | Strategy: {strategy}")
    print("=" * 69)

    if not os.path.exists(cuda_binary):
        compile_cuda_backend(out_binary=cuda_binary, arch_override=arch_override)

    assessor = GPUFaultAssessorV2(arch_override=arch_override)
    devfreq_path = assessor._devfreq_path

    from realistic_victims import create_victim
    victim_fn = create_victim(victim_type)

    # --- Collect platform info ---
    platform_info = collect_platform_info()
    platform_info["experiment"] = "oversubscription_P07"
    platform_info["seed"] = seed
    platform_info["victim_type"] = victim_type
    platform_info["strategy"] = strategy
    platform_info["num_trials"] = num_trials

    nvp_info = query_nvpmodel()
    if nvp_info:
        platform_info["nvpmodel_original"] = nvp_info
        print(f"[init] nvpmodel: {nvp_info.get('raw', 'unknown')}")

    nvp_modes = list_nvpmodel_modes()
    if nvp_modes:
        platform_info["nvpmodel_available"] = nvp_modes
        print(f"[init] nvpmodel modes available: {len(nvp_modes)}")
        for m in nvp_modes:
            print(f"        Mode {m['mode_id']}: {m['description']}")

    # --- Determine budget levels ---
    budget_levels = []

    if strategy == "devfreq" and devfreq_path:
        available_freqs = get_available_freqs(devfreq_path)
        orig_snap = devfreq_snapshot(devfreq_path)
        orig_max = int(orig_snap.get('max_freq', 0))

        if budget_freqs_mhz:
            for mhz in budget_freqs_mhz:
                target_hz = int(mhz * 1e6)
                closest = min(available_freqs, key=lambda f: abs(f - target_hz))
                budget_levels.append({
                    "label": f"{closest/1e6:.0f}MHz",
                    "freq_hz": closest,
                    "method": "devfreq",
                })
        else:
            freq_levels = compute_freq_budget_levels(available_freqs)
            for fhz in freq_levels:
                budget_levels.append({
                    "label": f"{fhz/1e6:.0f}MHz",
                    "freq_hz": fhz,
                    "method": "devfreq",
                })

        platform_info["devfreq_original"] = orig_snap
        platform_info["devfreq_available_freqs"] = available_freqs
        print(f"[init] devfreq path: {devfreq_path}")
        print(f"[init] Original max_freq: {orig_max/1e6:.0f} MHz")
        print(f"[init] Available: {[f/1e6 for f in available_freqs]} MHz")
        print(f"[init] Budget levels: {[b['label'] for b in budget_levels]}")

    elif strategy == "nvpmodel" and nvp_modes:
        orig_nvp = nvp_info
        for m in nvp_modes:
            budget_levels.append({
                "label": f"nvp_mode{m['mode_id']}",
                "mode_id": m["mode_id"],
                "freq_hz": 0,
                "method": "nvpmodel",
            })
        print(f"[init] Using nvpmodel modes: {[b['label'] for b in budget_levels]}")

    else:
        print("[!] No power-budget mechanism available. Trying devfreq fallback...")
        if devfreq_path:
            available_freqs = get_available_freqs(devfreq_path)
            orig_snap = devfreq_snapshot(devfreq_path)
            freq_levels = compute_freq_budget_levels(available_freqs)
            for fhz in freq_levels:
                budget_levels.append({
                    "label": f"{fhz/1e6:.0f}MHz",
                    "freq_hz": fhz,
                    "method": "devfreq",
                })
        if not budget_levels:
            raise RuntimeError(
                "No power budget mechanism available on this platform. "
                "Need devfreq sysfs or nvpmodel."
            )

    # Save platform info
    pinfo_path = os.path.join(output_dir, "oversubscription_platform_info.json")
    with open(pinfo_path, "w") as f:
        json.dump(platform_info, f, indent=2, default=str)
    print(f"[init] Platform info saved to {pinfo_path}")

    # --- Build trial schedule ---
    # For each trial: iterate budget levels, at each level run baseline then attack
    # Randomize budget-level order across trials for counterbalancing
    all_records = []
    all_summaries = []

    # Store original state for restoration
    orig_devfreq_snap = devfreq_snapshot(devfreq_path) if devfreq_path else None
    orig_nvp = nvp_info

    try:
        # Warmup
        print("\n[+] Warmup trial (discarded)...")
        if budget_levels and budget_levels[0]["method"] == "devfreq":
            run_oversubscription_trial(
                assessor, cuda_binary, victim_fn, devfreq_path,
                budget_levels[0]["freq_hz"], "warmup", duration_per_phase,
                trial_num=0, run_attacker=True,
            )
        print("[+] Warmup complete.\n")

        for trial_idx in range(1, num_trials + 1):
            print(f"\n{'=' * 69}")
            print(f"  Oversubscription Trial {trial_idx}/{num_trials}")
            print(f"{'=' * 69}")

            trial_order = list(budget_levels)
            random.shuffle(trial_order)

            for budget in trial_order:
                label = budget["label"]
                method = budget["method"]

                # Apply the power constraint
                if method == "devfreq" and devfreq_path:
                    freq_hz = budget["freq_hz"]
                    print(f"\n  [budget] Setting devfreq cap: {freq_hz/1e6:.0f} MHz")
                    set_devfreq_max(devfreq_path, freq_hz)
                    time.sleep(1)
                elif method == "nvpmodel":
                    mode_id = budget["mode_id"]
                    print(f"\n  [budget] Setting nvpmodel mode {mode_id}")
                    if not set_nvpmodel(mode_id):
                        print(f"    [!] Skipping mode {mode_id}")
                        continue
                    freq_hz = 0
                    if devfreq_path:
                        try:
                            freq_hz = int(_read_devfreq(devfreq_path, 'max_freq'))
                        except (IOError, OSError, ValueError):
                            pass
                    budget["freq_hz"] = freq_hz

                # Phase 1: Baseline (victim only, constrained)
                records_b, summary_b = run_oversubscription_trial(
                    assessor, cuda_binary, victim_fn, devfreq_path,
                    budget.get("freq_hz", 0), label, duration_per_phase,
                    trial_num=trial_idx, run_attacker=False,
                )
                summary_b["trial"] = trial_idx
                all_records.extend(records_b)
                all_summaries.append(summary_b)

                # Brief cooldown between conditions
                print(f"    [cooldown] {cooldown_between}s...")
                time.sleep(cooldown_between)

                # Phase 2: Attack + victim, constrained
                records_a, summary_a = run_oversubscription_trial(
                    assessor, cuda_binary, victim_fn, devfreq_path,
                    budget.get("freq_hz", 0), label, duration_per_phase,
                    trial_num=trial_idx, run_attacker=True,
                )
                summary_a["trial"] = trial_idx
                all_records.extend(records_a)
                all_summaries.append(summary_a)

                # Cooldown between budget levels
                print(f"    [cooldown] {cooldown_between}s before next budget level...")
                time.sleep(cooldown_between)

                # Restore between budget levels to avoid hysteresis
                if method == "devfreq" and devfreq_path and orig_devfreq_snap:
                    try:
                        devfreq_restore(devfreq_path, orig_devfreq_snap)
                    except (IOError, OSError):
                        pass
                    time.sleep(1)

    finally:
        # Restore original power settings
        print("\n[+] Restoring original power settings...")
        if orig_devfreq_snap and devfreq_path:
            try:
                devfreq_restore(devfreq_path, orig_devfreq_snap)
            except (IOError, OSError) as e:
                print(f"    [!] devfreq restore failed: {e}")
                print(f"    [!] Manual restore: write {orig_devfreq_snap} to {devfreq_path}")

        if strategy == "nvpmodel" and orig_nvp and orig_nvp.get("mode_id") is not None:
            try:
                set_nvpmodel(orig_nvp["mode_id"])
            except Exception as e:
                print(f"    [!] nvpmodel restore failed: {e}")
                print(f"    [!] Manual restore: sudo nvpmodel -m {orig_nvp['mode_id']}")

    # --- Write CSVs ---
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\n[+] Per-sample data saved to: {output_csv} ({len(all_records)} rows)")

    if all_summaries:
        sum_fields = list(all_summaries[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sum_fields)
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"[+] Summary saved to: {summary_csv}")

    # --- Print aggregate results ---
    _print_oversubscription_summary(all_summaries, budget_levels, num_trials)

    return all_summaries


def _print_oversubscription_summary(all_summaries, budget_levels, num_trials):
    """Print aggregate oversubscription results with throttling analysis."""
    print("\n" + "=" * 80)
    print("      PROJECT GAIA: OVERSUBSCRIPTION EXPERIMENT RESULTS (P07)")
    print(f"      {num_trials} trials per condition")
    print("=" * 80)

    labels = sorted(set(s["budget_label"] for s in all_summaries if s["budget_label"] != "warmup"))

    print(f"\n  {'Budget':<16}{'Condition':<12}{'Latency(ms)':<22}{'Power(W)':<18}{'MaxPow':<10}{'Cap%':<8}{'Clock(MHz)':<12}")
    print("  " + "-" * 96)

    for label in labels:
        for is_atk in [0, 1]:
            cond = "attack" if is_atk else "baseline"
            rows = [s for s in all_summaries
                    if s["budget_label"] == label and s["is_attack"] == is_atk]
            if not rows:
                continue

            lats = [r["mean_total_latency_ms"] for r in rows]
            pows = [r["mean_power_w"] for r in rows]
            max_pows = [r["max_power_w"] for r in rows]
            caps = [r["capping_rate_pct"] for r in rows]
            clks = [r["mean_clock_mhz"] for r in rows]

            lat_str = f"{np.mean(lats):.2f} ± {np.std(lats, ddof=1):.2f}" if len(lats) > 1 else f"{np.mean(lats):.2f}"
            pow_str = f"{np.mean(pows):.2f} ± {np.std(pows, ddof=1):.2f}" if len(pows) > 1 else f"{np.mean(pows):.2f}"

            print(f"  {label:<16}{cond:<12}{lat_str:<22}{pow_str:<18}"
                  f"{np.mean(max_pows):<10.1f}{np.mean(caps):<8.1f}{np.mean(clks):<12.0f}")

    # --- Oversubscription evidence summary ---
    print(f"\n  Oversubscription Evidence:")
    any_throttle = False
    for label in labels:
        atk_rows = [s for s in all_summaries
                     if s["budget_label"] == label and s["is_attack"] == 1]
        if not atk_rows:
            continue
        cap_rate = np.mean([r["capping_rate_pct"] for r in atk_rows])
        max_pow = np.max([r["max_power_w"] for r in atk_rows])
        freq_mhz = atk_rows[0].get("freq_cap_mhz", 0)

        if cap_rate > 0:
            any_throttle = True
            print(f"    [{label}] THROTTLING DETECTED: {cap_rate:.1f}% of samples "
                  f"(max power: {max_pow:.1f}W at {freq_mhz:.0f} MHz cap)")
        else:
            print(f"    [{label}] No throttling observed "
                  f"(max power: {max_pow:.1f}W at {freq_mhz:.0f} MHz cap)")

    # --- Attack amplification at each budget ---
    print(f"\n  Attack-vs-Baseline Amplification:")
    for label in labels:
        base_rows = [s for s in all_summaries
                      if s["budget_label"] == label and s["is_attack"] == 0]
        atk_rows = [s for s in all_summaries
                     if s["budget_label"] == label and s["is_attack"] == 1]
        if not base_rows or not atk_rows:
            continue

        base_lat = np.mean([r["mean_total_latency_ms"] for r in base_rows])
        atk_lat = np.mean([r["mean_total_latency_ms"] for r in atk_rows])
        base_pow = np.mean([r["mean_power_w"] for r in base_rows])
        atk_pow = np.mean([r["mean_power_w"] for r in atk_rows])

        lat_delta = atk_lat - base_lat
        lat_pct = (lat_delta / base_lat * 100) if base_lat > 0 else 0
        pow_delta = atk_pow - base_pow

        print(f"    [{label}] Latency: {base_lat:.2f} -> {atk_lat:.2f} ms "
              f"(+{lat_delta:.2f} ms, +{lat_pct:.1f}%) | "
              f"Power: {base_pow:.1f} -> {atk_pow:.1f} W (+{pow_delta:.1f} W)")

    if not any_throttle:
        print(f"\n  [!] WARNING: No throttling events observed at ANY budget level.")
        print(f"      The devfreq cap constrains clock but may not trigger sw_power_cap/hw_slowdown")
        print(f"      flags on Jetson. The POWER REDUCTION itself at lower clock caps is still")
        print(f"      evidence of a constrained budget — check whether the attack draws power")
        print(f"      up to the effective ceiling at each cap level.")
        print(f"      Consider: does victim degradation INCREASE at lower budgets? If so,")
        print(f"      that demonstrates oversubscription-like behavior even without the flag.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project GAIA: Oversubscription Experiment (Proposal P07)"
    )
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration per phase in seconds (default: 15)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per budget level (default: 5)")
    parser.add_argument("--arch", type=str, default=None,
                        help="GPU arch override (e.g. sm_110)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--victim-type", type=str, default="synthetic",
                        choices=["synthetic", "training", "inference"],
                        help="Victim workload type (default: synthetic)")
    parser.add_argument("--strategy", type=str, default="devfreq",
                        choices=["devfreq", "nvpmodel"],
                        help="Power budget mechanism (default: devfreq)")
    parser.add_argument("--budget", type=str, default=None,
                        help="Comma-separated clock caps in MHz (e.g. 1575,1200,900,630)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Cooldown between conditions in seconds (default: 5)")
    args = parser.parse_args()

    budget_freqs = None
    if args.budget:
        budget_freqs = [float(x.strip()) for x in args.budget.split(",")]

    run_oversubscription_experiment(
        duration_per_phase=args.duration,
        output_csv=args.out,
        num_trials=args.trials,
        arch_override=args.arch,
        seed=args.seed,
        victim_type=args.victim_type,
        budget_freqs_mhz=budget_freqs,
        strategy=args.strategy,
        cooldown_between=args.cooldown,
    )
