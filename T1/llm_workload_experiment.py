"""
Project GAIA: LLM Workload Phase Characterization Experiment
=============================================================
Addresses proposal commitment: "we will first model the phases of
power consumption in a GPU during LLM training and inference."

Runs GPT-2 Small (124M params) fine-tuning and text-generation inference
as victim workloads, measuring per-phase power telemetry under baseline
and attack conditions.

Captures the distinctive transformer training phases:
  - Forward pass (self-attention + FFN)
  - Backward pass (gradient computation)
  - Optimizer step (weight update)
  - Data loading / idle gaps

Usage (on Jetson via SSH):
    sudo python3 llm_workload_experiment.py --trials 5 --duration 15
    sudo python3 llm_workload_experiment.py --trials 5 --model gpt2
    sudo python3 llm_workload_experiment.py --trials 5 --victim-type inference
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
    _read_devfreq, devfreq_snapshot, devfreq_restore,
)

_EXE = ".exe" if sys.platform == "win32" else ""

# ---------------------------------------------------------------------------
# LLM victim workloads
# ---------------------------------------------------------------------------

_torch = None
_transformers = None


def _ensure_deps():
    """Import torch and transformers, with fallback guidance."""
    global _torch, _transformers
    if _torch is not None:
        return True
    try:
        import torch
        _torch = torch
    except ImportError:
        print("[!] PyTorch not found. Install via: pip3 install torch")
        return False
    try:
        import transformers
        _transformers = transformers
    except ImportError:
        print("[!] HuggingFace transformers not found. Install via:")
        print("    pip3 install transformers datasets")
        return False
    return True


def _get_device():
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    print("[!] CUDA not available, falling back to CPU (results will not reflect GPU power)")
    return _torch.device("cpu")


class LLMTrainingVictim:
    """GPT-2 Small fine-tuning victim with per-phase instrumentation."""

    def __init__(self, model_name="gpt2", seq_len=128, batch_size=4, use_fp16=True):
        if not _ensure_deps():
            raise RuntimeError("Missing dependencies for LLM victim")

        self.device = _get_device()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_name = model_name

        print(f"[llm-train] Loading {model_name} (seq_len={seq_len}, batch={batch_size}, fp16={self.use_fp16})...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.train()

        if self.use_fp16:
            self.model = self.model.half()

        self.optimizer = _torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.scaler = _torch.amp.GradScaler("cuda") if self.use_fp16 else None

        self._synthetic_batch = self._make_synthetic_batch()
        self._step_count = 0

        if self.device.type == "cuda":
            _torch.cuda.synchronize()

        print(f"[llm-train] Model loaded. Params: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

    def _make_synthetic_batch(self):
        """Create a fixed synthetic batch for reproducible training iterations."""
        input_ids = _torch.randint(
            0, self.tokenizer.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device, dtype=_torch.long,
        )
        attention_mask = _torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

    def __call__(self):
        """Run one training step. Returns timing dict with phase breakdown."""
        t0 = time.time()

        if self.device.type == "cuda":
            _torch.cuda.synchronize()

        # Forward pass
        t_fwd_start = time.time()
        if self.use_fp16:
            with _torch.amp.autocast("cuda"):
                outputs = self.model(**self._synthetic_batch)
                loss = outputs.loss
        else:
            outputs = self.model(**self._synthetic_batch)
            loss = outputs.loss

        if self.device.type == "cuda":
            _torch.cuda.synchronize()
        t_fwd_end = time.time()

        # Backward pass
        t_bwd_start = time.time()
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.device.type == "cuda":
            _torch.cuda.synchronize()
        t_bwd_end = time.time()

        # Optimizer step
        t_opt_start = time.time()
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.device.type == "cuda":
            _torch.cuda.synchronize()
        t_opt_end = time.time()

        t1 = time.time()
        self._step_count += 1

        tokens_processed = self.batch_size * self.seq_len

        return {
            "total_latency_ms": (t1 - t0) * 1000.0,
            "gpu_compute_ms": (t_fwd_end - t_fwd_start) * 1000.0,
            "forward_ms": (t_fwd_end - t_fwd_start) * 1000.0,
            "backward_ms": (t_bwd_end - t_bwd_start) * 1000.0,
            "optimizer_ms": (t_opt_end - t_opt_start) * 1000.0,
            "loss": float(loss.item()),
            "tokens_processed": tokens_processed,
            "tokens_per_sec": tokens_processed / max(t1 - t0, 1e-6),
            "step": self._step_count,
            "victim_backend": f"llm_train_{self.model_name}",
        }


class LLMInferenceVictim:
    """GPT-2 Small text generation victim."""

    def __init__(self, model_name="gpt2", prompt_len=32, gen_len=64, batch_size=2, use_fp16=True):
        if not _ensure_deps():
            raise RuntimeError("Missing dependencies for LLM victim")

        self.device = _get_device()
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model_name = model_name

        print(f"[llm-infer] Loading {model_name} (prompt={prompt_len}, gen={gen_len}, fp16={self.use_fp16})...")
        from transformers import GPT2LMHeadModel

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()

        self._prompt_ids = _torch.randint(
            0, self.model.config.vocab_size,
            (self.batch_size, self.prompt_len),
            device=self.device, dtype=_torch.long,
        )
        self._attention_mask = _torch.ones_like(self._prompt_ids)

        if self.device.type == "cuda":
            _torch.cuda.synchronize()

        print(f"[llm-infer] Model loaded for inference.")

    def __call__(self):
        """Run one generation pass. Returns timing dict."""
        t0 = time.time()

        if self.device.type == "cuda":
            _torch.cuda.synchronize()

        with _torch.no_grad():
            if self.use_fp16:
                with _torch.amp.autocast("cuda"):
                    outputs = self.model.generate(
                        input_ids=self._prompt_ids,
                        attention_mask=self._attention_mask,
                        max_new_tokens=self.gen_len,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=self.model.config.eos_token_id,
                    )
            else:
                outputs = self.model.generate(
                    input_ids=self._prompt_ids,
                    attention_mask=self._attention_mask,
                    max_new_tokens=self.gen_len,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.model.config.eos_token_id,
                )

        if self.device.type == "cuda":
            _torch.cuda.synchronize()

        t1 = time.time()

        tokens_generated = (outputs.shape[1] - self.prompt_len) * self.batch_size
        elapsed = max(t1 - t0, 1e-6)

        return {
            "total_latency_ms": elapsed * 1000.0,
            "gpu_compute_ms": elapsed * 1000.0,
            "forward_ms": elapsed * 1000.0,
            "backward_ms": 0.0,
            "optimizer_ms": 0.0,
            "loss": 0.0,
            "tokens_processed": tokens_generated,
            "tokens_per_sec": tokens_generated / elapsed,
            "step": 0,
            "victim_backend": f"llm_inference_{self.model_name}",
        }


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_llm_trial(
    assessor, cuda_binary, victim, devfreq_path,
    phase_name, duration_per_phase, trial_num,
    run_attacker=False,
):
    """Run one trial phase with LLM victim. Returns (records, summary)."""

    records = []
    trial_start = time.time()
    prev_current_a = 0.0
    prev_time = trial_start

    is_attack = 1 if run_attacker else 0
    category = f"llm_{'attack' if run_attacker else 'baseline'}"

    print(f"  [T{trial_num}] Phase '{phase_name}' (attacker: {run_attacker})...")

    cuda_proc = None
    if run_attacker:
        cmd = [
            cuda_binary,
            "--mode", "dc_stress",
            "--duration", str(duration_per_phase + 3.0),
        ]
        try:
            cuda_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=_cuda_subprocess_env(),
            )
            time.sleep(0.5)
        except Exception as e:
            print(f"    [!] Failed to spawn stress process: {e}")

    latencies_tot = []
    latencies_fwd = []
    latencies_bwd = []
    latencies_opt = []
    throughputs = []

    phase_start = time.time()

    while (time.time() - phase_start) < duration_per_phase:
        v_res = victim()
        latencies_tot.append(v_res["total_latency_ms"])
        latencies_fwd.append(v_res["forward_ms"])
        latencies_bwd.append(v_res["backward_ms"])
        latencies_opt.append(v_res["optimizer_ms"])
        throughputs.append(v_res["tokens_per_sec"])

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
            "victim_type": v_res.get("victim_backend", "llm"),
            "data_source": hw["provenance"],
            "power_w": round(p_w, 3),
            "voltage_v": round(v_meas, 5),
            "current_a": round(curr_a, 4),
            "di_dt_amps_per_sec": round(di_dt, 4),
            "victim_total_latency_ms": round(v_res["total_latency_ms"], 3),
            "forward_ms": round(v_res["forward_ms"], 3),
            "backward_ms": round(v_res["backward_ms"], 3),
            "optimizer_ms": round(v_res["optimizer_ms"], 3),
            "tokens_per_sec": round(v_res["tokens_per_sec"], 2),
            "loss": round(v_res.get("loss", 0.0), 4),
            "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
            "gpu_temp_c": hw.get("gpu_temp_c", 0),
            "gpu_util_pct": hw.get("gpu_util_pct"),
            "mem_util_pct": hw.get("mem_util_pct"),
            "sw_power_cap": hw.get("sw_power_cap", 0),
            "hw_slowdown": hw.get("hw_slowdown", 0),
            "devfreq_cur_hz": devfreq_cur,
        })

        prev_current_a = curr_a
        prev_time = t_curr

    if cuda_proc is not None:
        cuda_proc.terminate()
        try:
            cuda_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cuda_proc.kill()

    phase_records = [r for r in records if r["phase"] == phase_name]
    summary = {
        "phase": phase_name,
        "is_attack": is_attack,
        "trial": trial_num,
        "mean_total_latency_ms": float(np.mean(latencies_tot)),
        "std_total_latency_ms": float(np.std(latencies_tot, ddof=1)) if len(latencies_tot) > 1 else 0.0,
        "mean_forward_ms": float(np.mean(latencies_fwd)),
        "mean_backward_ms": float(np.mean(latencies_bwd)),
        "mean_optimizer_ms": float(np.mean(latencies_opt)),
        "mean_tokens_per_sec": float(np.mean(throughputs)),
        "std_tokens_per_sec": float(np.std(throughputs, ddof=1)) if len(throughputs) > 1 else 0.0,
        "mean_power_w": float(np.mean([r["power_w"] for r in phase_records])),
        "std_power_w": float(np.std([r["power_w"] for r in phase_records], ddof=1)) if len(phase_records) > 1 else 0.0,
        "max_power_w": float(np.max([r["power_w"] for r in phase_records])),
        "mean_clock_mhz": float(np.mean([r["sm_clock_mhz"] for r in phase_records if r["sm_clock_mhz"] > 0])) if any(r["sm_clock_mhz"] > 0 for r in phase_records) else 0.0,
        "n_samples": len(phase_records),
        "n_steps": len(latencies_tot),
    }

    print(f"    Latency: {summary['mean_total_latency_ms']:.2f} +/- {summary['std_total_latency_ms']:.2f} ms | "
          f"Fwd: {summary['mean_forward_ms']:.2f} | Bwd: {summary['mean_backward_ms']:.2f} | Opt: {summary['mean_optimizer_ms']:.2f} ms")
    print(f"    Throughput: {summary['mean_tokens_per_sec']:.1f} tok/s | "
          f"Power: {summary['mean_power_w']:.2f} W (max {summary['max_power_w']:.1f}) | "
          f"Clock: {summary['mean_clock_mhz']:.0f} MHz")

    return records, summary


def run_llm_experiment(
    cuda_binary=None,
    duration_per_phase=15.0,
    output_csv=None,
    num_trials=5,
    arch_override=None,
    seed=None,
    victim_type="training",
    model_name="gpt2",
    seq_len=128,
    batch_size=4,
    cooldown_between=5.0,
):
    """Run the full LLM workload experiment."""

    if cuda_binary is None:
        cuda_binary = os.path.join(SCRIPT_DIR, "gpu_power_poc_v3" + _EXE)
    if output_csv is None:
        output_csv = os.path.join(
            SCRIPT_DIR, "..", "results", "llm",
            f"llm_{victim_type}.csv",
        )

    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = output_csv.replace(".csv", "_summary.csv")

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    print("=" * 69)
    print(" Project GAIA: LLM Workload Phase Experiment")
    print(f" Model: {model_name} | Victim: {victim_type}")
    print(f" Trials: {num_trials} | Duration/phase: {duration_per_phase}s | Seed: {seed}")
    print(f" Seq len: {seq_len} | Batch size: {batch_size}")
    print("=" * 69)

    if not os.path.exists(cuda_binary):
        compile_cuda_backend(out_binary=cuda_binary, arch_override=arch_override)

    assessor = GPUFaultAssessorV2(arch_override=arch_override)
    devfreq_path = assessor._devfreq_path

    # Create victim
    if victim_type == "training":
        victim = LLMTrainingVictim(
            model_name=model_name,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    elif victim_type == "inference":
        victim = LLMInferenceVictim(
            model_name=model_name,
            prompt_len=min(seq_len, 32),
            gen_len=min(seq_len, 64),
            batch_size=max(batch_size // 2, 1),
        )
    else:
        raise ValueError(f"Unknown victim_type: {victim_type}. Use 'training' or 'inference'.")

    # Platform info
    platform_info = collect_platform_info()
    platform_info["experiment"] = "llm_workload_phase"
    platform_info["seed"] = seed
    platform_info["model_name"] = model_name
    platform_info["victim_type"] = victim_type
    platform_info["seq_len"] = seq_len
    platform_info["batch_size"] = batch_size
    platform_info["num_trials"] = num_trials

    pinfo_path = os.path.join(output_dir, f"llm_{victim_type}_platform_info.json")
    with open(pinfo_path, "w") as f:
        json.dump(platform_info, f, indent=2, default=str)
    print(f"[init] Platform info saved to {pinfo_path}")

    all_records = []
    all_summaries = []

    # Warmup
    print("\n[+] Warmup (5s, discarded)...")
    for _ in range(3):
        victim()
    print("[+] Warmup complete.\n")

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'=' * 69}")
        print(f"  LLM Trial {trial_idx}/{num_trials}")
        print(f"{'=' * 69}")

        # Phase 1: Baseline (victim only)
        recs, summ = run_llm_trial(
            assessor, cuda_binary, victim, devfreq_path,
            phase_name=f"llm_{victim_type}_baseline",
            duration_per_phase=duration_per_phase,
            trial_num=trial_idx,
            run_attacker=False,
        )
        all_records.extend(recs)
        all_summaries.append(summ)

        print(f"    [cooldown] {cooldown_between}s...")
        time.sleep(cooldown_between)

        # Phase 2: Attack + victim
        recs, summ = run_llm_trial(
            assessor, cuda_binary, victim, devfreq_path,
            phase_name=f"llm_{victim_type}_attack",
            duration_per_phase=duration_per_phase,
            trial_num=trial_idx,
            run_attacker=True,
        )
        all_records.extend(recs)
        all_summaries.append(summ)

        print(f"    [cooldown] {cooldown_between}s...")
        time.sleep(cooldown_between)

    # Write CSVs
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

    # Print aggregate results
    _print_llm_summary(all_summaries, victim_type, num_trials)

    return all_summaries


def _print_llm_summary(all_summaries, victim_type, num_trials):
    """Print aggregate LLM experiment results."""
    print("\n" + "=" * 80)
    print(f"      PROJECT GAIA: LLM WORKLOAD EXPERIMENT RESULTS")
    print(f"      Victim: {victim_type} | {num_trials} trials")
    print("=" * 80)

    for is_atk in [0, 1]:
        cond = "ATTACK" if is_atk else "BASELINE"
        rows = [s for s in all_summaries if s["is_attack"] == is_atk]
        if not rows:
            continue

        lats = [r["mean_total_latency_ms"] for r in rows]
        pows = [r["mean_power_w"] for r in rows]
        tputs = [r["mean_tokens_per_sec"] for r in rows]
        fwds = [r["mean_forward_ms"] for r in rows]
        bwds = [r["mean_backward_ms"] for r in rows]
        opts = [r["mean_optimizer_ms"] for r in rows]

        print(f"\n  [{cond}]")
        print(f"    Latency:    {np.mean(lats):.2f} +/- {np.std(lats, ddof=1):.2f} ms" if len(lats) > 1 else f"    Latency:    {np.mean(lats):.2f} ms")
        print(f"    Throughput: {np.mean(tputs):.1f} +/- {np.std(tputs, ddof=1):.1f} tok/s" if len(tputs) > 1 else f"    Throughput: {np.mean(tputs):.1f} tok/s")
        print(f"    Power:      {np.mean(pows):.2f} +/- {np.std(pows, ddof=1):.2f} W" if len(pows) > 1 else f"    Power:      {np.mean(pows):.2f} W")
        if victim_type == "training":
            print(f"    Phase breakdown: Fwd {np.mean(fwds):.2f} | Bwd {np.mean(bwds):.2f} | Opt {np.mean(opts):.2f} ms")

    # Attack impact
    base_rows = [s for s in all_summaries if s["is_attack"] == 0]
    atk_rows = [s for s in all_summaries if s["is_attack"] == 1]
    if base_rows and atk_rows:
        base_lat = np.mean([r["mean_total_latency_ms"] for r in base_rows])
        atk_lat = np.mean([r["mean_total_latency_ms"] for r in atk_rows])
        base_tput = np.mean([r["mean_tokens_per_sec"] for r in base_rows])
        atk_tput = np.mean([r["mean_tokens_per_sec"] for r in atk_rows])
        base_pow = np.mean([r["mean_power_w"] for r in base_rows])
        atk_pow = np.mean([r["mean_power_w"] for r in atk_rows])

        lat_pct = (atk_lat - base_lat) / base_lat * 100 if base_lat > 0 else 0
        tput_pct = (atk_tput - base_tput) / base_tput * 100 if base_tput > 0 else 0
        pow_delta = atk_pow - base_pow

        print(f"\n  ATTACK IMPACT:")
        print(f"    Latency:    +{lat_pct:.1f}% ({base_lat:.2f} -> {atk_lat:.2f} ms)")
        print(f"    Throughput: {tput_pct:+.1f}% ({base_tput:.1f} -> {atk_tput:.1f} tok/s)")
        print(f"    Power:      +{pow_delta:.1f} W ({base_pow:.1f} -> {atk_pow:.1f} W)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project GAIA: LLM Workload Phase Experiment"
    )
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration per phase in seconds (default: 15)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials (default: 5)")
    parser.add_argument("--arch", type=str, default=None,
                        help="GPU arch override (e.g. sm_110)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--victim-type", type=str, default="training",
                        choices=["training", "inference"],
                        help="LLM victim type (default: training)")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for training (default: 128)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Cooldown between conditions (default: 5s)")
    args = parser.parse_args()

    run_llm_experiment(
        duration_per_phase=args.duration,
        output_csv=args.out,
        num_trials=args.trials,
        arch_override=args.arch,
        seed=args.seed,
        victim_type=args.victim_type,
        model_name=args.model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        cooldown_between=args.cooldown,
    )
