"""
Project GAIA: Analysis of Cross-Examiner Experiments (Nov 2026)
===============================================================
Produces publication figures, statistical tables, and LaTeX fragments
for three experiments:
  1. T1 Oversubscription (P07) — power-budget sweep
  2. T1 Contention Isolation (G03) — matched-load design
  3. T3 Blinded Attribution (P15) — closed-loop adversary ID

Usage:
    python analyze_new_results.py <results_dir> [--out-dir figures/]

Where <results_dir> contains:
    latest/t1_oversubscription.csv
    contention_isolation/contention_isolation.csv
    T3/blinded_attribution.csv
    T3/blinded_attribution_attribution_log.json
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
from scipy.stats import t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def ci95(values):
    vals = np.array(values, dtype=float)
    n = len(vals)
    if n < 2:
        m = float(np.mean(vals))
        return m, 0.0, m, m
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1)) / np.sqrt(n)
    tc = t_dist.ppf(0.975, df=n - 1)
    margin = tc * se
    return m, margin, m - margin, m + margin


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sp


# =====================================================================
#  1. OVERSUBSCRIPTION ANALYSIS (P07)
# =====================================================================
def analyze_oversubscription(df, out_dir):
    print("\n" + "=" * 70)
    print("  T1: OVERSUBSCRIPTION / POWER-BUDGET SWEEP (P07)")
    print("=" * 70)

    budgets = sorted(df["freq_cap_mhz"].unique())
    print(f"  Budget levels: {budgets}")
    print(f"  Trials: {df['trial'].nunique()}")

    rows = []
    for cap in budgets:
        sub = df[df["freq_cap_mhz"] == cap]
        base = sub[sub["is_attack"] == 0]["power_w"]
        atk = sub[sub["is_attack"] == 1]["power_w"]
        base_lat = sub[sub["is_attack"] == 0]["victim_total_latency_ms"]
        atk_lat = sub[sub["is_attack"] == 1]["victim_total_latency_ms"]

        bm, bci, _, _ = ci95(base)
        am, aci, _, _ = ci95(atk)
        overhead_w = am - bm
        ratio = am / bm if bm > 0 else 0
        tdp_pct = am / 130.0 * 100

        blm, _, _, _ = ci95(base_lat)
        alm, _, _, _ = ci95(atk_lat)
        lat_pct = (alm - blm) / blm * 100 if blm > 0 else 0

        rows.append({
            "freq_cap_mhz": int(cap),
            "baseline_power_w": round(bm, 1),
            "attack_power_w": round(am, 1),
            "overhead_w": round(overhead_w, 1),
            "ratio": round(ratio, 2),
            "tdp_pct": round(tdp_pct, 1),
            "baseline_lat_ms": round(blm, 1),
            "attack_lat_ms": round(alm, 1),
            "lat_degradation_pct": round(lat_pct, 1),
        })

        print(f"\n  Clock cap: {int(cap)} MHz")
        print(f"    Baseline power: {bm:.1f} +/- {bci:.1f} W")
        print(f"    Attack power:   {am:.1f} +/- {aci:.1f} W  "
              f"(+{overhead_w:.1f}W, {ratio:.2f}x, {tdp_pct:.0f}% TDP)")
        print(f"    Latency: {blm:.1f} -> {alm:.1f} ms ({lat_pct:+.1f}%)")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(out_dir, "oversubscription_analysis.csv"), index=False)

    # --- Figure: Power by budget level ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    x = np.arange(len(budgets))
    base_means = [r["baseline_power_w"] for r in rows]
    atk_means = [r["attack_power_w"] for r in rows]
    labels = [f"{int(b)}" for b in budgets]

    w = 0.35
    ax1.bar(x - w/2, base_means, w, label="Baseline", color="#4878CF", alpha=0.85)
    ax1.bar(x + w/2, atk_means, w, label="Under attack", color="#D65F5F", alpha=0.85)
    ax1.axhline(130, color="gray", linestyle="--", linewidth=0.8, label="TDP (130W)")
    ax1.set_xlabel("Clock cap (MHz)")
    ax1.set_ylabel("Mean power (W)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=7)
    ax1.set_title("(a) Power vs. clock budget")

    overheads = [r["overhead_w"] for r in rows]
    ax2.bar(x, overheads, color="#6ACC65", alpha=0.85)
    ax2.set_xlabel("Clock cap (MHz)")
    ax2.set_ylabel("Power overhead (W)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("(b) Attack-induced overhead")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_oversubscription.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_oversubscription.png"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_oversubscription.pdf")

    # --- LaTeX table ---
    latex_lines = []
    latex_lines.append("% Auto-generated by analyze_new_results.py")
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering\\small")
    latex_lines.append("\\caption{Power-budget oversubscription sweep (5 trials). "
                       "The attacker forces significant power overhead at every "
                       "clock budget level.}")
    latex_lines.append("\\label{tab:oversubscription}")
    latex_lines.append("\\begin{tabular}{rrrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Clock cap & Baseline & Attack & Overhead & Ratio & TDP\\% \\\\")
    latex_lines.append("(MHz) & (W) & (W) & (W) & & \\\\")
    latex_lines.append("\\midrule")
    for r in rows:
        latex_lines.append(
            f"{r['freq_cap_mhz']} & {r['baseline_power_w']:.1f} & "
            f"{r['attack_power_w']:.1f} & +{r['overhead_w']:.1f} & "
            f"{r['ratio']:.2f}$\\times$ & {r['tdp_pct']:.0f}\\% \\\\"
        )
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    tex_path = os.path.join(out_dir, "tab_oversubscription.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  LaTeX table saved: {tex_path}")

    return summary_df


# =====================================================================
#  2. CONTENTION ISOLATION ANALYSIS (G03)
# =====================================================================
def analyze_contention_isolation(df, out_dir):
    print("\n" + "=" * 70)
    print("  T1: CONTENTION ISOLATION / MATCHED-LOAD DESIGN (G03)")
    print("=" * 70)

    conditions = df["condition"].unique()
    clock_policies = df["clock_policy"].unique()
    print(f"  Conditions: {list(conditions)}")
    print(f"  Clock policies: {list(clock_policies)}")
    print(f"  Trials: {df['trial'].nunique()}")

    rows = []
    for cond in sorted(conditions):
        for policy in sorted(clock_policies):
            sub = df[(df["condition"] == cond) & (df["clock_policy"] == policy)]
            if sub.empty:
                continue

            base = sub[sub["is_attack"] == 0]
            atk = sub[sub["is_attack"] == 1]

            if base.empty or atk.empty:
                continue

            bp, _, _, _ = ci95(base["power_w"])
            ap, _, _, _ = ci95(atk["power_w"])
            bl, _, _, _ = ci95(base["victim_total_latency_ms"])
            al, _, _, _ = ci95(atk["victim_total_latency_ms"])

            power_delta = ap - bp
            lat_delta_pct = (al - bl) / bl * 100 if bl > 0 else 0

            rows.append({
                "condition": cond,
                "clock_policy": policy,
                "baseline_power_w": round(bp, 2),
                "attack_power_w": round(ap, 2),
                "power_delta_w": round(power_delta, 2),
                "baseline_lat_ms": round(bl, 2),
                "attack_lat_ms": round(al, 2),
                "lat_degradation_pct": round(lat_delta_pct, 1),
            })

    iso_df = pd.DataFrame(rows)
    iso_df.to_csv(os.path.join(out_dir, "contention_isolation_analysis.csv"), index=False)

    for _, r in iso_df.iterrows():
        print(f"  {r['condition']:30s} [{r['clock_policy']:7s}]  "
              f"dP={r['power_delta_w']:+7.1f}W  "
              f"dL={r['lat_degradation_pct']:+6.1f}%")

    # Separate oversubscribe vs compute-only pairs
    oversub = iso_df[iso_df["condition"].str.contains("oversubscribe")]
    compute = iso_df[iso_df["condition"].str.contains("compute_only")]
    cache = iso_df[iso_df["condition"].str.contains("cache_only")]

    if not oversub.empty and not compute.empty:
        o_lat = oversub["lat_degradation_pct"].values
        c_lat = compute["lat_degradation_pct"].values
        print(f"\n  Oversubscribe latency degradation: {np.mean(o_lat):.1f}% "
              f"(n={len(o_lat)})")
        print(f"  Compute-only latency degradation:  {np.mean(c_lat):.1f}% "
              f"(n={len(c_lat)})")
        diff = np.mean(o_lat) - np.mean(c_lat)
        print(f"  Power-specific excess degradation:  {diff:+.1f}%")

    # DVFS coupling: compare default vs fixed clock policy
    default_rows = iso_df[iso_df["clock_policy"] == "default"]
    fixed_rows = iso_df[iso_df["clock_policy"] == "fixed"]
    if not default_rows.empty and not fixed_rows.empty:
        d_power = default_rows["power_delta_w"].mean()
        f_power = fixed_rows["power_delta_w"].mean()
        print(f"\n  DVFS coupling analysis:")
        print(f"    Default governor dP: {d_power:+.1f}W")
        print(f"    Fixed clock dP:      {f_power:+.1f}W")
        print(f"    DVFS-mediated:       {d_power - f_power:+.1f}W "
              f"({abs(d_power - f_power) / abs(f_power) * 100:.0f}% of fixed effect)")

    # --- Figure: Contention isolation grouped bar ---
    cond_labels_short = {
        "pair1_oversubscribe": "Oversub.",
        "pair2_compute_only": "Compute",
        "pair3_cache_only": "Cache",
        "pair4_mixed_stress": "Mixed",
        "pair5_pulsed_modulation": "Pulsed",
        "pair6_occupancy": "Occupancy",
        "pair7_thermal_sync": "Thermal",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    conds_ordered = sorted(iso_df["condition"].unique())
    x = np.arange(len(conds_ordered))
    width = 0.35

    for i, policy in enumerate(["default", "fixed"]):
        sub = iso_df[iso_df["clock_policy"] == policy]
        vals = []
        for c in conds_ordered:
            row = sub[sub["condition"] == c]
            vals.append(float(row["power_delta_w"].iloc[0]) if not row.empty else 0)
        offset = -width/2 if i == 0 else width/2
        color = "#4878CF" if policy == "default" else "#D65F5F"
        label = f"{policy} governor"
        ax1.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax1.set_ylabel("Power delta (W)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([cond_labels_short.get(c, c[:10]) for c in conds_ordered],
                        rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=7)
    ax1.set_title("(a) Power change by condition")
    ax1.axhline(0, color="gray", linewidth=0.5)

    for i, policy in enumerate(["default", "fixed"]):
        sub = iso_df[iso_df["clock_policy"] == policy]
        vals = []
        for c in conds_ordered:
            row = sub[sub["condition"] == c]
            vals.append(float(row["lat_degradation_pct"].iloc[0]) if not row.empty else 0)
        offset = -width/2 if i == 0 else width/2
        color = "#4878CF" if policy == "default" else "#D65F5F"
        ax2.bar(x + offset, vals, width, label=f"{policy} governor",
                color=color, alpha=0.85)

    ax2.set_ylabel("Latency degradation (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([cond_labels_short.get(c, c[:10]) for c in conds_ordered],
                        rotation=45, ha="right", fontsize=7)
    ax2.legend(fontsize=7)
    ax2.set_title("(b) Victim latency impact")
    ax2.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_contention_isolation.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_contention_isolation.png"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_contention_isolation.pdf")

    # --- LaTeX table ---
    latex_lines = []
    latex_lines.append("% Auto-generated by analyze_new_results.py")
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering\\small")
    latex_lines.append("\\caption{Contention isolation: matched-load experiment separating "
                       "power-domain effects from compute contention. "
                       "$\\Delta P$ and $\\Delta L$ are attack-induced changes.}")
    latex_lines.append("\\label{tab:contention_isolation}")
    latex_lines.append("\\begin{tabular}{llrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Condition & Clock & Baseline (W) & Attack (W) & "
                       "$\\Delta P$ (W) & $\\Delta L$ (\\%) \\\\")
    latex_lines.append("\\midrule")
    for _, r in iso_df.iterrows():
        cname = cond_labels_short.get(r["condition"], r["condition"])
        latex_lines.append(
            f"{cname} & {r['clock_policy']} & {r['baseline_power_w']:.1f} & "
            f"{r['attack_power_w']:.1f} & {r['power_delta_w']:+.1f} & "
            f"{r['lat_degradation_pct']:+.1f} \\\\"
        )
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table*}")

    tex_path = os.path.join(out_dir, "tab_contention_isolation.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  LaTeX table saved: {tex_path}")

    return iso_df


# =====================================================================
#  3. BLINDED ATTRIBUTION ANALYSIS (P15)
# =====================================================================
def analyze_blinded_attribution(df, attr_log, out_dir):
    print("\n" + "=" * 70)
    print("  T3: BLINDED ATTRIBUTION (P15)")
    print("=" * 70)

    n_trials = len(attr_log)
    correct = sum(1 for a in attr_log if a.get("correct_attribution"))
    accuracy = correct / n_trials * 100 if n_trials > 0 else 0

    latencies = [a["attribution_latency_s"] for a in attr_log]
    innocents = [a["innocent_kills"] for a in attr_log]
    candidates = [a["candidates_found"] for a in attr_log]

    lat_m, lat_ci, _, _ = ci95(latencies)
    inn_m, inn_ci, _, _ = ci95(innocents)

    print(f"  Trials: {n_trials}")
    print(f"  Attribution accuracy: {correct}/{n_trials} ({accuracy:.0f}%)")
    print(f"  Attribution latency:  {lat_m:.2f} +/- {lat_ci:.2f} s")
    print(f"  Innocent kills/trial: {inn_m:.1f} +/- {inn_ci:.1f}")
    print(f"  Candidates/trial:     {np.mean(candidates):.1f}")

    # Phase power analysis
    phases = ["multi_baseline", "multi_attack_undetected",
              "multi_attack_blinded_response"]
    print(f"\n  Phase power summary:")
    phase_stats = {}
    for phase in phases:
        sub = df[df["phase"] == phase]
        if sub.empty:
            continue
        pm, pci, _, _ = ci95(sub["power_w"])
        phase_stats[phase] = (pm, pci)
        print(f"    {phase:35s}: {pm:.1f} +/- {pci:.1f} W")

    if "multi_baseline" in phase_stats and "multi_attack_undetected" in phase_stats:
        bp = phase_stats["multi_baseline"][0]
        ap = phase_stats["multi_attack_undetected"][0]
        print(f"\n  Attack-induced power change: {ap - bp:+.1f}W "
              f"({(ap - bp) / bp * 100:+.1f}%)")
        print(f"  (Attack REDUCES power due to DVFS contention from duty-cycled kernel)")

    # Detection metrics
    det_phases = df[df["phase"].isin(["multi_attack_undetected",
                                       "multi_attack_blinded_response"])]
    if not det_phases.empty and "anomaly_detected" in det_phases.columns:
        total_attack_samples = len(det_phases)
        detected_samples = det_phases["anomaly_detected"].sum()
        det_rate = detected_samples / total_attack_samples * 100
        print(f"\n  Detection rate (attack samples): {detected_samples}/{total_attack_samples} "
              f"({det_rate:.1f}%)")

    baseline_powers = [a["baseline_power_w"] for a in attr_log]
    detection_powers = [a["detection_power_w"] for a in attr_log]
    recovery_magnitudes = [b - d for b, d in zip(baseline_powers, detection_powers)]
    rm, rci, _, _ = ci95(recovery_magnitudes)
    print(f"  Baseline -> detection power shift: -{rm:.1f} +/- {rci:.1f} W")

    # --- Figure: Attribution summary ---
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

    # (a) Power by phase
    ax = axes[0]
    phase_labels = ["Baseline", "Attack\n(undetected)", "Attack\n(response)"]
    phase_keys = phases
    powers = []
    power_cis = []
    for pk in phase_keys:
        sub = df[df["phase"] == pk]
        if not sub.empty:
            m, c, _, _ = ci95(sub["power_w"])
            powers.append(m)
            power_cis.append(c)
        else:
            powers.append(0)
            power_cis.append(0)

    colors = ["#4878CF", "#D65F5F", "#E5AE38"]
    ax.bar(range(len(phase_labels)), powers, yerr=power_cis,
           color=colors, alpha=0.85, capsize=3)
    ax.set_xticks(range(len(phase_labels)))
    ax.set_xticklabels(phase_labels, fontsize=7)
    ax.set_ylabel("Power (W)")
    ax.set_title("(a) Phase power")

    # (b) Attribution accuracy
    ax = axes[1]
    ax.bar(["Correct", "Incorrect"],
           [correct, n_trials - correct],
           color=["#6ACC65", "#D65F5F"], alpha=0.85)
    ax.set_ylabel("Trials")
    ax.set_title(f"(b) Attribution ({accuracy:.0f}%)")
    ax.set_ylim(0, max(n_trials + 1, 6))

    # (c) Attribution latency
    ax = axes[2]
    ax.bar(range(n_trials), latencies, color="#4878CF", alpha=0.85)
    ax.axhline(lat_m, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"(c) Latency ({lat_m:.1f}s avg)")
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels([str(i+1) for i in range(n_trials)])

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_blinded_attribution.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_blinded_attribution.png"))
    plt.close(fig)
    print(f"\n  Figure saved: fig_blinded_attribution.pdf")

    # --- Figure: Power timeseries for one trial ---
    trial1 = df[df["trial"] == 1].copy()
    if not trial1.empty:
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        ax.plot(trial1["timestamp"], trial1["power_w"],
                linewidth=0.5, color="#333333", alpha=0.7)

        for phase, color in [("multi_baseline", "#4878CF"),
                             ("multi_attack_undetected", "#D65F5F"),
                             ("multi_attack_blinded_response", "#E5AE38"),
                             ("multi_recovery", "#6ACC65")]:
            sub = trial1[trial1["phase"] == phase]
            if not sub.empty:
                ax.axvspan(sub["timestamp"].min(), sub["timestamp"].max(),
                          alpha=0.15, color=color, label=phase.replace("multi_", ""))

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Power (W)")
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.set_title("Trial 1: Power over time with phase annotations")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_attribution_timeseries.pdf"))
        fig.savefig(os.path.join(out_dir, "fig_attribution_timeseries.png"))
        plt.close(fig)
        print(f"  Figure saved: fig_attribution_timeseries.pdf")

    # --- LaTeX table ---
    latex_lines = []
    latex_lines.append("% Auto-generated by analyze_new_results.py")
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering\\small")
    latex_lines.append("\\caption{Blinded adversary attribution via sequential "
                       "elimination (5 trials, 3 concurrent GPU processes). "
                       "Baseline-recovery scoring identifies the attacker by "
                       "measuring which process termination moves system power "
                       "closest to the pre-attack baseline.}")
    latex_lines.append("\\label{tab:blinded_attribution}")
    latex_lines.append("\\begin{tabular}{lr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Metric & Value \\\\")
    latex_lines.append("\\midrule")
    latex_lines.append(f"Attribution accuracy & {correct}/{n_trials} "
                       f"({accuracy:.0f}\\%) \\\\")
    latex_lines.append(f"Attribution latency & {lat_m:.1f} $\\pm$ {lat_ci:.1f}\\,s \\\\")
    latex_lines.append(f"Innocent kills / trial & {inn_m:.1f} $\\pm$ {inn_ci:.1f} \\\\")
    latex_lines.append(f"Candidates / trial & {np.mean(candidates):.1f} \\\\")
    bp_m, bp_ci, _, _ = ci95(baseline_powers)
    dp_m, dp_ci, _, _ = ci95(detection_powers)
    latex_lines.append(f"Baseline power & {bp_m:.1f} $\\pm$ {bp_ci:.1f}\\,W \\\\")
    latex_lines.append(f"Detection power & {dp_m:.1f} $\\pm$ {dp_ci:.1f}\\,W \\\\")
    latex_lines.append(f"Attack power shift & $-${rm:.1f}\\,W ({rm/bp_m*100:.0f}\\%) \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    tex_path = os.path.join(out_dir, "tab_blinded_attribution.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  LaTeX table saved: {tex_path}")

    return {
        "accuracy": accuracy,
        "n_trials": n_trials,
        "correct": correct,
        "latency_mean": lat_m,
        "latency_ci": lat_ci,
        "innocent_kills_mean": inn_m,
    }


# =====================================================================
#  COMBINED SUMMARY
# =====================================================================
def print_combined_summary(oversub_df, iso_df, attr_stats):
    print("\n" + "=" * 70)
    print("  COMBINED EXPERIMENT SUMMARY FOR PAPER")
    print("=" * 70)

    if oversub_df is not None:
        max_cap = oversub_df.loc[oversub_df["freq_cap_mhz"].idxmax()]
        min_cap = oversub_df.loc[oversub_df["freq_cap_mhz"].idxmin()]
        print(f"\n  P07 (Oversubscription):")
        print(f"    At max clock ({int(max_cap['freq_cap_mhz'])} MHz): "
              f"+{max_cap['overhead_w']:.0f}W overhead, "
              f"{max_cap['ratio']:.2f}x baseline")
        print(f"    At min clock ({int(min_cap['freq_cap_mhz'])} MHz): "
              f"+{min_cap['overhead_w']:.0f}W overhead, "
              f"{min_cap['ratio']:.2f}x baseline")
        print(f"    Attacker forces oversubscription at ALL budget levels")

    if iso_df is not None:
        oversub_rows = iso_df[iso_df["condition"].str.contains("oversubscribe")]
        compute_rows = iso_df[iso_df["condition"].str.contains("compute_only")]
        if not oversub_rows.empty and not compute_rows.empty:
            o_mean = oversub_rows["lat_degradation_pct"].mean()
            c_mean = compute_rows["lat_degradation_pct"].mean()
            print(f"\n  G03 (Contention Isolation):")
            print(f"    Oversubscribe degradation: {o_mean:+.1f}%")
            print(f"    Compute-only degradation:  {c_mean:+.1f}%")
            print(f"    Power-specific excess:     {o_mean - c_mean:+.1f}%")

    if attr_stats is not None:
        print(f"\n  P15 (Blinded Attribution):")
        print(f"    Accuracy: {attr_stats['correct']}/{attr_stats['n_trials']} "
              f"({attr_stats['accuracy']:.0f}%)")
        print(f"    Latency:  {attr_stats['latency_mean']:.1f}s")
        print(f"    Method:   Sequential elimination with baseline-recovery scoring")


# =====================================================================
#  MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Project GAIA: Analyze cross-examiner experiment results")
    parser.add_argument("results_dir",
                        help="Directory containing experiment CSVs")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for figures/tables (default: results_dir/figures)")
    args = parser.parse_args()

    rdir = args.results_dir
    out_dir = args.out_dir or os.path.join(rdir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  PROJECT GAIA: CROSS-EXAMINER EXPERIMENT ANALYSIS")
    print(f"  Results dir: {rdir}")
    print(f"  Output dir:  {out_dir}")
    print("=" * 70)

    # Load CSVs
    oversub_path = os.path.join(rdir, "latest", "t1_oversubscription.csv")
    iso_path = os.path.join(rdir, "contention_isolation", "contention_isolation.csv")
    attr_csv_path = os.path.join(rdir, "T3", "blinded_attribution.csv")
    attr_log_path = os.path.join(rdir, "T3",
                                  "blinded_attribution_attribution_log.json")

    oversub_df = None
    iso_df = None
    attr_stats = None

    # 1. Oversubscription
    if os.path.exists(oversub_path):
        print(f"\n  Loading: {oversub_path}")
        df = pd.read_csv(oversub_path)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
        oversub_df = analyze_oversubscription(df, out_dir)
    else:
        print(f"\n  [skip] {oversub_path} not found")

    # 2. Contention isolation
    if os.path.exists(iso_path):
        print(f"\n  Loading: {iso_path}")
        df = pd.read_csv(iso_path)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
        iso_result = analyze_contention_isolation(df, out_dir)
    else:
        print(f"\n  [skip] {iso_path} not found")
        iso_result = None

    # 3. Blinded attribution
    if os.path.exists(attr_csv_path) and os.path.exists(attr_log_path):
        print(f"\n  Loading: {attr_csv_path}")
        df = pd.read_csv(attr_csv_path)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
        with open(attr_log_path) as f:
            attr_log = json.load(f)
        print(f"    Attribution log: {len(attr_log)} entries")
        attr_stats = analyze_blinded_attribution(df, attr_log, out_dir)
    else:
        print(f"\n  [skip] T3 attribution files not found")

    print_combined_summary(oversub_df, iso_result, attr_stats)

    print(f"\n  All outputs saved to: {out_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
