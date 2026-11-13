"""
Project GAIA: LLM Phase Analysis
=================================
Analyzes LLM workload experiment results and compares attack impact
across all victim types (matmul, CNN train, CNN inference, LLM train,
LLM inference).

Runs on Windows — reads CSVs produced by llm_workload_experiment.py.

Outputs:
  - paper/figures/fig_llm_phases.pdf: LLM training phase power profile
  - paper/figures/fig_llm_attack_impact.pdf: Cross-workload attack comparison
  - paper/figures/tab_llm_workload_comparison.tex: LaTeX table
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PROJECT_DIR, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_llm_data(results_dir):
    """Load LLM training and inference CSVs."""
    data = {}

    train_csv = os.path.join(results_dir, "llm", "llm_training.csv")
    if os.path.exists(train_csv):
        data["llm_training"] = pd.read_csv(train_csv)
        print(f"[+] LLM training: {len(data['llm_training'])} rows from {train_csv}")
    else:
        print(f"[!] LLM training CSV not found: {train_csv}")

    infer_csv = os.path.join(results_dir, "llm", "llm_inference.csv")
    if os.path.exists(infer_csv):
        data["llm_inference"] = pd.read_csv(infer_csv)
        print(f"[+] LLM inference: {len(data['llm_inference'])} rows from {infer_csv}")
    else:
        print(f"[!] LLM inference CSV not found: {infer_csv}")

    # Also load summary CSVs
    train_sum = os.path.join(results_dir, "llm", "llm_training_summary.csv")
    if os.path.exists(train_sum):
        data["llm_training_summary"] = pd.read_csv(train_sum)

    infer_sum = os.path.join(results_dir, "llm", "llm_inference_summary.csv")
    if os.path.exists(infer_sum):
        data["llm_inference_summary"] = pd.read_csv(infer_sum)

    return data


def analyze_llm_phases(data, out_dir):
    """Generate LLM phase power profile figure."""

    df = data.get("llm_training")
    if df is None:
        print("[!] No LLM training data, skipping phase analysis")
        return

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=False)

    # Panel (a): Power timeseries for baseline vs attack (single trial)
    ax = axes[0]
    for is_atk, label, color, ls in [(0, "Baseline", "#2166ac", "-"), (1, "Attack", "#b2182b", "--")]:
        mask = (df["is_attack"] == is_atk) & (df["trial"] == 1)
        subset = df[mask].copy()
        if subset.empty:
            continue
        t0 = subset["timestamp"].iloc[0]
        ax.plot(subset["timestamp"] - t0, subset["power_w"],
                color=color, ls=ls, lw=1.2, alpha=0.8, label=label)

    ax.set_ylabel("Module power (W)")
    ax.set_title("(a) LLM training power profile (Trial 1)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    # Panel (b): Phase breakdown comparison
    ax = axes[1]
    summary = data.get("llm_training_summary")
    if summary is not None and "mean_forward_ms" in summary.columns:
        base = summary[summary["is_attack"] == 0]
        atk = summary[summary["is_attack"] == 1]

        phases = ["Forward", "Backward", "Optimizer"]
        base_means = [
            base["mean_forward_ms"].mean(),
            base["mean_backward_ms"].mean(),
            base["mean_optimizer_ms"].mean(),
        ]
        atk_means = [
            atk["mean_forward_ms"].mean(),
            atk["mean_backward_ms"].mean(),
            atk["mean_optimizer_ms"].mean(),
        ]

        x = np.arange(len(phases))
        w = 0.35
        ax.bar(x - w/2, base_means, w, label="Baseline", color="#2166ac", alpha=0.8)
        ax.bar(x + w/2, atk_means, w, label="Attack", color="#b2182b", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("(b) Per-phase latency breakdown", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "Phase breakdown data not available",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_llm_phases.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[+] Saved {out_path}")


def analyze_cross_workload(data, out_dir, existing_results_dir=None):
    """Generate cross-workload attack comparison figure and table.

    Combines LLM results with existing matmul/CNN results from T1/T3.
    If existing results are available, includes them in the comparison.
    """

    workloads = []

    # LLM training
    llm_train_sum = data.get("llm_training_summary")
    if llm_train_sum is not None:
        base = llm_train_sum[llm_train_sum["is_attack"] == 0]
        atk = llm_train_sum[llm_train_sum["is_attack"] == 1]
        if not base.empty and not atk.empty:
            b_lat = base["mean_total_latency_ms"].mean()
            a_lat = atk["mean_total_latency_ms"].mean()
            b_pow = base["mean_power_w"].mean()
            a_pow = atk["mean_power_w"].mean()
            b_tput = base["mean_tokens_per_sec"].mean()
            a_tput = atk["mean_tokens_per_sec"].mean()
            workloads.append({
                "name": "LLM Training",
                "short": "LLM Train",
                "baseline_lat": b_lat,
                "attack_lat": a_lat,
                "lat_pct": (a_lat - b_lat) / b_lat * 100,
                "baseline_pow": b_pow,
                "attack_pow": a_pow,
                "pow_delta": a_pow - b_pow,
                "baseline_tput": b_tput,
                "attack_tput": a_tput,
                "tput_pct": (a_tput - b_tput) / b_tput * 100 if b_tput > 0 else 0,
            })

    # LLM inference
    llm_infer_sum = data.get("llm_inference_summary")
    if llm_infer_sum is not None:
        base = llm_infer_sum[llm_infer_sum["is_attack"] == 0]
        atk = llm_infer_sum[llm_infer_sum["is_attack"] == 1]
        if not base.empty and not atk.empty:
            b_lat = base["mean_total_latency_ms"].mean()
            a_lat = atk["mean_total_latency_ms"].mean()
            b_pow = base["mean_power_w"].mean()
            a_pow = atk["mean_power_w"].mean()
            b_tput = base["mean_tokens_per_sec"].mean()
            a_tput = atk["mean_tokens_per_sec"].mean()
            workloads.append({
                "name": "LLM Inference",
                "short": "LLM Infer",
                "baseline_lat": b_lat,
                "attack_lat": a_lat,
                "lat_pct": (a_lat - b_lat) / b_lat * 100,
                "baseline_pow": b_pow,
                "attack_pow": a_pow,
                "pow_delta": a_pow - b_pow,
                "baseline_tput": b_tput,
                "attack_tput": a_tput,
                "tput_pct": (a_tput - b_tput) / b_tput * 100 if b_tput > 0 else 0,
            })

    # Existing workloads from the paper (hardcoded from Table 2 in the paper)
    existing = [
        {"name": "Matrix Multiply", "short": "MatMul",
         "baseline_lat": 3.57, "attack_lat": 6.86, "lat_pct": 92,
         "baseline_pow": 46.9, "attack_pow": 86.0, "pow_delta": 39.1,
         "baseline_tput": 0, "attack_tput": 0, "tput_pct": 0},
        {"name": "CNN Training", "short": "CNN Train",
         "baseline_lat": 3.30, "attack_lat": 4.35, "lat_pct": 32,
         "baseline_pow": 46.9, "attack_pow": 86.0, "pow_delta": 39.1,
         "baseline_tput": 0, "attack_tput": 0, "tput_pct": 0},
        {"name": "CNN Inference", "short": "CNN Infer",
         "baseline_lat": 1.80, "attack_lat": 1.87, "lat_pct": 4,
         "baseline_pow": 46.9, "attack_pow": 86.0, "pow_delta": 39.1,
         "baseline_tput": 0, "attack_tput": 0, "tput_pct": 0},
    ]

    all_workloads = existing + workloads

    if not workloads:
        print("[!] No LLM results available for cross-workload comparison")
        print("[!] Run the LLM experiment on the Jetson first, then re-run this analysis")
        return

    # Figure: latency degradation bar chart
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel (a): Latency degradation %
    ax = axes[0]
    names = [w["short"] for w in all_workloads]
    lat_pcts = [w["lat_pct"] for w in all_workloads]
    colors = ["#2166ac"] * len(existing) + ["#b2182b"] * len(workloads)
    bars = ax.bar(range(len(all_workloads)), lat_pcts, color=colors, alpha=0.8)
    ax.set_xticks(range(len(all_workloads)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Latency degradation (%)")
    ax.set_title("(a) Attack impact by workload", fontsize=9)
    for bar, pct in zip(bars, lat_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"+{pct:.0f}%", ha="center", va="bottom", fontsize=7)

    # Panel (b): Power overhead
    ax = axes[1]
    pow_deltas = [w["pow_delta"] for w in all_workloads]
    bars = ax.bar(range(len(all_workloads)), pow_deltas, color=colors, alpha=0.8)
    ax.set_xticks(range(len(all_workloads)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Power overhead (W)")
    ax.set_title("(b) Power injection by workload", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_llm_attack_impact.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[+] Saved {out_path}")

    # LaTeX table
    tex_path = os.path.join(out_dir, "tab_llm_workload_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\\small\n")
        f.write("\\caption{Attack impact across all victim workload types (5 trials,\n")
        f.write("  sustained stress). LLM workloads use GPT-2 Small (124M) with FP16.}\n")
        f.write("\\label{tab:llm_workload}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Workload & Base lat & Atk lat & Degrad. & Power \\\\\n")
        f.write("         & (ms)     & (ms)    & (\\%)    & delta (W) \\\\\n")
        f.write("\\midrule\n")
        for w in all_workloads:
            f.write(f"{w['name']} & {w['baseline_lat']:.2f} & {w['attack_lat']:.2f} "
                    f"& $+${w['lat_pct']:.0f}\\% & $+${w['pow_delta']:.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[+] Saved {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="GAIA LLM Phase Analysis")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory containing llm/ subfolder")
    parser.add_argument("--out-dir", type=str, default=FIG_DIR,
                        help="Output directory for figures/tables")
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        candidates = [
            os.path.join(PROJECT_DIR, "11.06.2026_results", "results"),
            os.path.join(PROJECT_DIR, "results"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                results_dir = c
                break
        if results_dir is None:
            results_dir = os.path.join(PROJECT_DIR, "results")

    print(f"[+] Results directory: {results_dir}")
    print(f"[+] Output directory: {args.out_dir}")

    data = load_llm_data(results_dir)

    if not data:
        print("[!] No LLM data found. Run llm_workload_experiment.py on the Jetson first.")
        print("[!] Then copy results to: <results_dir>/llm/")
        return

    analyze_llm_phases(data, args.out_dir)
    analyze_cross_workload(data, args.out_dir)

    print("\n[+] LLM analysis complete.")


if __name__ == "__main__":
    main()
