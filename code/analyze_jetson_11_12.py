"""Analyze Jetson 11/12/2026 results: filler kernel defense + LLM workload."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import mannwhitneyu

FONT = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
PAGE_WIDTH = 7.16

STYLE = {
    'font.family': FONT, 'font.size': 9,
    'axes.labelcolor': '#202020', 'text.color': '#202020',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .6, 'lines.linewidth': 1.6,
    'axes.formatter.useoffset': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .35, 'grid.color': '#D0D4DA', 'grid.linewidth': .4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 600, 'legend.frameon': False,
}

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "jetson_11_12_2026"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(STYLE)


# ── Filler Kernel Defense ──────────────────────────────────────────

def analyze_filler_kernel():
    df = pd.read_csv(RESULTS / "filler_kernel" / "filler_defense.csv")
    print(f"Filler kernel: {len(df)} rows, {df['trial'].nunique()} trials")

    conditions = [
        ("victim_alone", "Victim only", "#555555"),
        ("victim_attacker", "Victim + attacker", "#D55E00"),
        ("victim_attacker_fill25", "Victim + attacker\n+ 25% filler", "#E69F00"),
        ("victim_attacker_fill50", "Victim + attacker\n+ 50% filler", "#0072B2"),
        ("victim_filler_only25", "Victim + 25% filler", "#009E73"),
        ("victim_filler_only50", "Victim + 50% filler", "#56B4E9"),
    ]

    # Per-trial means for error bars
    trial_means = df.groupby(["trial", "condition"])["total_power_w"].mean().reset_index()

    means, stds, labels, colors = [], [], [], []
    for cond, label, color in conditions:
        subset = trial_means[trial_means["condition"] == cond]["total_power_w"]
        means.append(subset.mean())
        stds.append(subset.std())
        labels.append(label)
        colors.append(color)

    # ── Figure 1: Bar chart of power by condition ──
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.48, 2.8), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.65, error_kw={'linewidth': 0.8})
    ax.set_ylabel("Mean power (W)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha='center')
    ax.set_ylim(0, 110)

    # Annotate reduction
    atk_power = means[1]  # victim_attacker
    base_power = means[0]  # victim_alone
    overhead = atk_power - base_power
    for i, (cond, _, _) in enumerate(conditions):
        if "fill" in cond and "only" not in cond:
            reduction = atk_power - means[i]
            pct = reduction / overhead * 100
            ax.annotate(f"−{reduction:.0f}W\n({pct:.0f}%)",
                        xy=(x[i], means[i] + stds[i] + 1), fontsize=6.5,
                        ha='center', va='bottom', color=colors[i], weight='bold')

    ax.axhline(base_power, color='#555555', ls='--', lw=0.6, alpha=0.5)
    ax.axhline(atk_power, color='#D55E00', ls='--', lw=0.6, alpha=0.5)

    fig.savefig(FIGURES / "fig_filler_defense.pdf")
    print(f"  Saved: {FIGURES / 'fig_filler_defense.pdf'}")
    plt.close(fig)

    # ── Statistical tests ──
    for cond_name in ["victim_attacker_fill25", "victim_attacker_fill50"]:
        atk = trial_means[trial_means["condition"] == "victim_attacker"]["total_power_w"]
        fill = trial_means[trial_means["condition"] == cond_name]["total_power_w"]
        U, p = mannwhitneyu(atk, fill, alternative='greater')
        print(f"  {cond_name} vs attacker: U={U}, p={p:.4f}")

    # ── LaTeX table ──
    tab_lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Filler kernel defense: mean power and victim latency across 5 trials.}",
        r"  \label{tab:filler_defense}",
        r"  \small",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Condition & Power (W) & Latency (ms) & $\Delta$P (W) \\",
        r"    \midrule",
    ]
    for cond, label, _ in conditions:
        sub = df[df["condition"] == cond]
        pw = sub["total_power_w"]
        lat = sub["total_latency_ms"]
        delta = pw.mean() - base_power
        delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        label_clean = label.replace("\n", " ").replace("%", r"\%")
        tab_lines.append(
            f"    {label_clean} & {pw.mean():.1f} $\\pm$ {pw.std():.1f} "
            f"& {lat.mean():.2f} $\\pm$ {lat.std():.2f} & {delta_str} \\\\"
        )
    tab_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tab_path = FIGURES / "tab_filler_defense.tex"
    tab_path.write_text("\n".join(tab_lines))
    print(f"  Saved: {tab_path}")


# ── LLM Workload ──────────────────────────────────────────────────

def analyze_llm():
    df_train = pd.read_csv(RESULTS / "llm_training" / "llm_training.csv")
    df_infer = pd.read_csv(RESULTS / "llm_inference" / "llm_inference.csv")
    print(f"LLM training: {len(df_train)} rows, inference: {len(df_infer)} rows")

    # ── Figure 2: Grouped bar — training vs inference, baseline vs attack ──
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 2.4), constrained_layout=True)

    datasets = {
        "Training": df_train,
        "Inference": df_infer,
    }

    metrics = [
        ("power_w", "Power (W)", axes[0]),
        ("victim_total_latency_ms", "Latency (ms)", axes[1]),
        ("tokens_per_sec", "Throughput (tok/s)", axes[2]),
    ]

    bar_w = 0.3
    colors_ba = ["#0072B2", "#D55E00"]

    for metric_col, ylabel, ax in metrics:
        x = np.arange(len(datasets))
        for j, phase_label in enumerate(["baseline", "attack"]):
            vals, errs = [], []
            for ds_name, ds_df in datasets.items():
                phase_key = f"llm_{'training' if 'Training' in ds_name else 'inference'}_{phase_label}"
                trial_means = ds_df[ds_df["phase"] == phase_key].groupby("trial")[metric_col].mean()
                vals.append(trial_means.mean())
                errs.append(trial_means.std())
            ax.bar(x + j * bar_w - bar_w / 2, vals, bar_w, yerr=errs,
                   capsize=3, color=colors_ba[j], edgecolor='white', linewidth=0.5,
                   label=phase_label.capitalize() if metric_col == "power_w" else None,
                   error_kw={'linewidth': 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(list(datasets.keys()), fontsize=8)
        ax.set_ylabel(ylabel)

    axes[0].legend(loc='upper left', fontsize=7)

    fig.savefig(FIGURES / "fig_llm_workload.pdf")
    print(f"  Saved: {FIGURES / 'fig_llm_workload.pdf'}")
    plt.close(fig)

    # ── Figure 3: Training phase breakdown (forward/backward/optimizer) ──
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.48, 2.4), constrained_layout=True)

    phases_cols = [("forward_ms", "Forward", "#0072B2"),
                   ("backward_ms", "Backward", "#E69F00"),
                   ("optimizer_ms", "Optimizer", "#D55E00")]

    x = np.arange(2)
    labels_x = ["Baseline", "Under attack"]
    bottoms = np.zeros(2)

    for col, label, color in phases_cols:
        vals = []
        for phase_key in ["llm_training_baseline", "llm_training_attack"]:
            trial_means = df_train[df_train["phase"] == phase_key].groupby("trial")[col].mean()
            vals.append(trial_means.mean())
        ax.bar(x, vals, 0.5, bottom=bottoms, label=label, color=color, edgecolor='white', linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.legend(loc='upper left', fontsize=7)
    ax.set_title("GPT-2 training step breakdown")

    fig.savefig(FIGURES / "fig_llm_phase_breakdown.pdf")
    print(f"  Saved: {FIGURES / 'fig_llm_phase_breakdown.pdf'}")
    plt.close(fig)

    # ── LaTeX table ──
    tab_lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{LLM workload attack impact on Jetson AGX Thor (GPT-2, 124M params, 5 trials).}",
        r"  \label{tab:llm_workload}",
        r"  \small",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Workload & Condition & Power (W) & Latency (ms) & Throughput (tok/s) \\",
        r"    \midrule",
    ]

    for ds_name, ds_df in datasets.items():
        wl = ds_name.lower()
        for phase_label in ["baseline", "attack"]:
            phase_key = f"llm_{wl}_{phase_label}"
            sub = ds_df[ds_df["phase"] == phase_key]
            pw = sub.groupby("trial")["power_w"].mean()
            lat = sub.groupby("trial")["victim_total_latency_ms"].mean()
            tps = sub.groupby("trial")["tokens_per_sec"].mean()
            tab_lines.append(
                f"    {ds_name} & {phase_label.capitalize()} "
                f"& {pw.mean():.1f} $\\pm$ {pw.std():.1f} "
                f"& {lat.mean():.1f} $\\pm$ {lat.std():.1f} "
                f"& {tps.mean():.0f} $\\pm$ {tps.std():.0f} \\\\"
            )
        if ds_name == "Training":
            tab_lines.append(r"    \midrule")

    tab_lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tab_path = FIGURES / "tab_llm_workload.tex"
    tab_path.write_text("\n".join(tab_lines))
    print(f"  Saved: {tab_path}")

    # Print attack impact summary
    for ds_name, ds_df in datasets.items():
        wl = ds_name.lower()
        base = ds_df[ds_df["phase"] == f"llm_{wl}_baseline"]
        atk = ds_df[ds_df["phase"] == f"llm_{wl}_attack"]
        lat_b = base.groupby("trial")["victim_total_latency_ms"].mean().mean()
        lat_a = atk.groupby("trial")["victim_total_latency_ms"].mean().mean()
        pw_b = base.groupby("trial")["power_w"].mean().mean()
        pw_a = atk.groupby("trial")["power_w"].mean().mean()
        tps_b = base.groupby("trial")["tokens_per_sec"].mean().mean()
        tps_a = atk.groupby("trial")["tokens_per_sec"].mean().mean()
        print(f"\n  {ds_name}:")
        print(f"    Latency: {lat_b:.1f} -> {lat_a:.1f} ms (+{(lat_a/lat_b-1)*100:.1f}%)")
        print(f"    Power:   {pw_b:.1f} -> {pw_a:.1f} W ({pw_a-pw_b:+.1f}W)")
        print(f"    Throughput: {tps_b:.0f} -> {tps_a:.0f} tok/s ({(tps_a/tps_b-1)*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("GAIA Jetson 11/12/2026 Results Analysis")
    print("=" * 60)
    analyze_filler_kernel()
    print()
    analyze_llm()
    print("\nDone.")
