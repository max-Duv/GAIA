"""PCA analysis of GPU telemetry for attack/baseline discrimination."""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

OVERSUB_CSV = os.path.join(
    "11.06.2026_results", "results", "latest", "t1_oversubscription.csv"
)
CONTENTION_CSV = os.path.join(
    "11.06.2026_results", "results", "contention_isolation", "contention_isolation.csv"
)
OUT_DIR = os.path.join("paper", "figures")

FEATURES = [
    "power_w", "voltage_v", "current_a", "di_dt_amps_per_sec",
    "sm_clock_mhz", "gpu_temp_c", "victim_total_latency_ms",
]

FEATURE_LABELS = {
    "power_w": "Power (W)",
    "voltage_v": "Voltage (V)",
    "current_a": "Current (A)",
    "di_dt_amps_per_sec": "di/dt (A/s)",
    "sm_clock_mhz": "SM Clock (MHz)",
    "gpu_temp_c": "GPU Temp (C)",
    "victim_total_latency_ms": "Latency (ms)",
}


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if x.size < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) if cov[0, 0] > 0 and cov[1, 1] > 0 else 0
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def load_and_prepare():
    df_os = pd.read_csv(OVERSUB_CSV)
    df_ci = pd.read_csv(CONTENTION_CSV)

    df_os["dataset"] = "oversubscription"
    df_os["condition_label"] = df_os["budget_label"] + " " + df_os["phase"].str.extract(r"(baseline|attack)", expand=False)
    df_ci["dataset"] = "contention_isolation"
    df_ci["condition_label"] = df_ci["condition"] + " (" + df_ci["clock_policy"] + ")"

    shared_cols = FEATURES + ["is_attack", "dataset", "condition_label"]
    df_os_sub = df_os[[c for c in shared_cols if c in df_os.columns]].copy()
    df_ci_sub = df_ci[[c for c in shared_cols if c in df_ci.columns]].copy()

    df = pd.concat([df_os_sub, df_ci_sub], ignore_index=True)
    df = df.dropna(subset=FEATURES)

    # Drop zero-variance features
    for f in FEATURES[:]:
        if df[f].std() < 1e-10:
            print(f"Dropping zero-variance feature: {f}")
            FEATURES.remove(f)

    return df


def run_pca(df):
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)
    return pca, scores, scaler


def fig_scree(pca, path):
    n = len(pca.explained_variance_ratio_)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    indvar = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(range(1, n + 1), indvar, color="#4878CF", alpha=0.7, label="Individual")
    ax.plot(range(1, n + 1), cumvar, "o-", color="#D65F5F", markersize=5, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xticks(range(1, n + 1))
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scree plot: {path}")


def fig_scatter(df, scores, pca, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    # Left panel: oversubscription data
    mask_os = df["dataset"] == "oversubscription"
    ax = axes[0]
    for is_atk, color, label in [(0, "#4878CF", "Baseline"), (1, "#D65F5F", "Attack")]:
        m = mask_os & (df["is_attack"] == is_atk)
        idx = m.values
        ax.scatter(scores[idx, 0], scores[idx, 1], c=color, alpha=0.3, s=10, label=label, rasterized=True)
        confidence_ellipse(scores[idx, 0], scores[idx, 1], ax, n_std=2.0,
                           edgecolor=color, facecolor="none", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("Oversubscription", fontsize=10)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right panel: contention isolation data
    mask_ci = df["dataset"] == "contention_isolation"
    ax = axes[1]
    for is_atk, color, label in [(0, "#4878CF", "Baseline"), (1, "#D65F5F", "Attack")]:
        m = mask_ci & (df["is_attack"] == is_atk)
        idx = m.values
        ax.scatter(scores[idx, 0], scores[idx, 1], c=color, alpha=0.3, s=10, label=label, rasterized=True)
        confidence_ellipse(scores[idx, 0], scores[idx, 1], ax, n_std=2.0,
                           edgecolor=color, facecolor="none", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("Contention Isolation", fontsize=10)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {path}")


def latex_table(pca, path):
    n_pcs = min(3, len(pca.explained_variance_ratio_))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PCA variance decomposition and dominant feature loadings.}",
        r"\label{tab:pca_variance}",
        r"\small",
        r"\begin{tabular}{clcc}",
        r"\toprule",
        r"\textbf{PC} & \textbf{Top Loadings} & \textbf{Var. (\%)} & \textbf{Cum. (\%)} \\",
        r"\midrule",
    ]

    for i in range(n_pcs):
        loadings = pca.components_[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        loading_strs = []
        for j in top_idx:
            fname = FEATURE_LABELS.get(FEATURES[j], FEATURES[j])
            loading_strs.append(f"{fname} ({loadings[j]:+.2f})")
        var_pct = pca.explained_variance_ratio_[i] * 100
        row = f"PC{i+1} & {'; '.join(loading_strs)} & {var_pct:.1f} & {cumvar[i]:.1f} \\\\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table: {path}")


def print_summary(pca):
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print("\n--- PCA Summary ---")
    for i, (v, c) in enumerate(zip(pca.explained_variance_ratio_ * 100, cumvar)):
        print(f"  PC{i+1}: {v:.1f}% (cumulative: {c:.1f}%)")
    print(f"  Components for 95% variance: {np.searchsorted(cumvar, 95.0) + 1}")
    print()
    for i in range(min(3, len(pca.components_))):
        loadings = pca.components_[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        parts = [f"{FEATURES[j]}={loadings[j]:+.3f}" for j in top_idx]
        print(f"  PC{i+1} loadings: {', '.join(parts)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading data...")
    df = load_and_prepare()
    print(f"  Combined dataset: {len(df)} samples, {len(FEATURES)} features")
    print(f"  Features: {FEATURES}")
    print(f"  Attack samples: {(df['is_attack']==1).sum()}, Baseline: {(df['is_attack']==0).sum()}")

    print("Running PCA...")
    pca, scores, scaler = run_pca(df)

    print_summary(pca)

    fig_scree(pca, os.path.join(OUT_DIR, "fig_pca_scree.pdf"))
    fig_scatter(df, scores, pca, os.path.join(OUT_DIR, "fig_pca_scatter.pdf"))
    latex_table(pca, os.path.join(OUT_DIR, "tab_pca_variance.tex"))

    print("\nDone.")


if __name__ == "__main__":
    main()
