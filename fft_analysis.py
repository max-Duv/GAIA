"""
FFT spectral analysis of GAIA power telemetry.
Computes frequency-domain features from INA3221 power traces to distinguish
attack conditions from baseline workloads.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, stats

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "11.06.2026_results", "results")
FIG_DIR = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def compute_psd(power_trace, fs):
    """Welch PSD estimate. Returns frequencies and power spectral density."""
    nperseg = min(len(power_trace), 64)
    freqs, psd = signal.welch(power_trace, fs=fs, nperseg=nperseg,
                              noverlap=nperseg // 2, scaling="density")
    return freqs, psd


def spectral_features(freqs, psd):
    """Extract spectral features from a PSD."""
    total_energy = np.trapezoid(psd, freqs)

    # Discrete normalization for entropy (sum to 1)
    psd_discrete = psd[1:] / (np.sum(psd[1:]) + 1e-12)  # skip DC
    spectral_entropy = -np.sum(psd_discrete * np.log2(psd_discrete + 1e-12))

    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-12)
    peak_idx = np.argmax(psd[1:]) + 1  # skip DC
    dominant_freq = freqs[peak_idx]
    peak_power = psd[peak_idx]

    # Band energies: low (0-0.5Hz), mid (0.5-1.0Hz), high (1.0-Nyquist)
    low_mask = (freqs > 0) & (freqs <= 0.5)
    mid_mask = (freqs > 0.5) & (freqs <= 1.0)
    high_mask = freqs > 1.0
    low_energy = np.trapezoid(psd[low_mask], freqs[low_mask]) if low_mask.any() else 0
    mid_energy = np.trapezoid(psd[mid_mask], freqs[mid_mask]) if mid_mask.any() else 0
    high_energy = np.trapezoid(psd[high_mask], freqs[high_mask]) if high_mask.any() else 0

    return {
        "total_energy": total_energy,
        "spectral_entropy": spectral_entropy,
        "spectral_centroid": spectral_centroid,
        "dominant_freq_hz": dominant_freq,
        "peak_psd": peak_power,
        "low_band_energy": low_energy,
        "mid_band_energy": mid_energy,
        "high_band_energy": high_energy,
    }


def estimate_fs(timestamps):
    """Estimate sampling frequency from timestamps."""
    diffs = np.diff(timestamps)
    return 1.0 / np.median(diffs)


def analyze_oversubscription_fft():
    """FFT analysis of oversubscription experiment: baseline vs attack at each clock cap."""
    csv_path = os.path.join(RESULTS, "latest", "t1_oversubscription.csv")
    df = pd.read_csv(csv_path)

    budget_labels = sorted(df["budget_label"].unique(),
                           key=lambda x: int(x.replace("MHz", "")))

    all_features = []
    psd_curves = {}

    for blabel in budget_labels:
        for is_atk in [0, 1]:
            cond_label = "Attack" if is_atk else "Baseline"
            sub = df[(df["budget_label"] == blabel) & (df["is_attack"] == is_atk)]
            # Aggregate across trials for PSD
            trial_psds = []
            for trial_id, trial_df in sub.groupby("trial"):
                trial_df = trial_df.sort_values("timestamp")
                power = trial_df["power_w"].values
                ts = trial_df["timestamp"].values
                if len(power) < 16:
                    continue
                fs = estimate_fs(ts)
                freqs, psd = compute_psd(power, fs)
                trial_psds.append((freqs, psd))

            if not trial_psds:
                continue

            # Interpolate all PSDs to common frequency grid before averaging
            max_len = max(len(p[0]) for p in trial_psds)
            ref_freqs = trial_psds[np.argmax([len(p[0]) for p in trial_psds])][0]
            interp_psds = []
            for f, p in trial_psds:
                if len(f) == len(ref_freqs):
                    interp_psds.append(p)
                else:
                    interp_psds.append(np.interp(ref_freqs, f, p))
            avg_psd = np.mean(interp_psds, axis=0)
            feats = spectral_features(ref_freqs, avg_psd)
            feats["budget"] = blabel
            feats["condition"] = cond_label
            feats["freq_cap_mhz"] = int(blabel.replace("MHz", ""))
            all_features.append(feats)
            psd_curves[(blabel, cond_label)] = (ref_freqs, avg_psd)

    return pd.DataFrame(all_features), psd_curves, budget_labels


def analyze_contention_fft():
    """FFT analysis of contention isolation: fixed vs default clock, baseline vs attack."""
    csv_path = os.path.join(RESULTS, "contention_isolation", "contention_isolation.csv")
    df = pd.read_csv(csv_path)

    all_features = []
    psd_curves = {}

    for policy in ["fixed", "default"]:
        for is_atk in [0, 1]:
            cond_label = "Attack" if is_atk else "Baseline"
            sub = df[(df["clock_policy"] == policy) & (df["is_attack"] == is_atk)]
            trial_psds = []
            for trial_id, trial_df in sub.groupby("trial"):
                trial_df = trial_df.sort_values("timestamp")
                power = trial_df["power_w"].values
                ts = trial_df["timestamp"].values
                if len(power) < 16:
                    continue
                fs = estimate_fs(ts)
                freqs, psd = compute_psd(power, fs)
                trial_psds.append((freqs, psd))

            if not trial_psds:
                continue

            ref_freqs = trial_psds[np.argmax([len(p[0]) for p in trial_psds])][0]
            interp_psds = []
            for f, p in trial_psds:
                if len(f) == len(ref_freqs):
                    interp_psds.append(p)
                else:
                    interp_psds.append(np.interp(ref_freqs, f, p))
            avg_psd = np.mean(interp_psds, axis=0)
            feats = spectral_features(ref_freqs, avg_psd)
            feats["clock_policy"] = policy
            feats["condition"] = cond_label
            all_features.append(feats)
            psd_curves[(policy, cond_label)] = (ref_freqs, avg_psd)

    return pd.DataFrame(all_features), psd_curves


def make_psd_figure(oversub_curves, budget_labels, contention_curves):
    """Two-panel PSD figure: (a) oversubscription at representative caps, (b) contention isolation."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Panel (a): Oversubscription — pick 3 representative caps (low/mid/high)
    representative = [budget_labels[0], budget_labels[len(budget_labels)//2], budget_labels[-1]]
    colors_base = ["#2166ac", "#4393c3", "#92c5de"]
    colors_atk = ["#b2182b", "#d6604d", "#f4a582"]

    ax = axes[0]
    for i, blabel in enumerate(representative):
        key_b = (blabel, "Baseline")
        key_a = (blabel, "Attack")
        if key_b in oversub_curves:
            f, p = oversub_curves[key_b]
            ax.semilogy(f[1:], p[1:], color=colors_base[i], linewidth=1.2,
                       label=f"{blabel} base", linestyle="-")
        if key_a in oversub_curves:
            f, p = oversub_curves[key_a]
            ax.semilogy(f[1:], p[1:], color=colors_atk[i], linewidth=1.2,
                       label=f"{blabel} atk", linestyle="--")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (W$^2$/Hz)")
    ax.set_title("(a) Power-budget oversubscription")
    ax.legend(ncol=2, fontsize=6, frameon=False, loc="upper right")

    # Panel (b): Contention isolation
    ax = axes[1]
    style_map = {
        ("fixed", "Baseline"): ("#2166ac", "-", "Fixed base"),
        ("fixed", "Attack"): ("#b2182b", "--", "Fixed atk"),
        ("default", "Baseline"): ("#4393c3", "-", "Default base"),
        ("default", "Attack"): ("#d6604d", "--", "Default atk"),
    }
    for key, (color, ls, label) in style_map.items():
        if key in contention_curves:
            f, p = contention_curves[key]
            ax.semilogy(f[1:], p[1:], color=color, linewidth=1.2,
                       label=label, linestyle=ls)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (W$^2$/Hz)")
    ax.set_title("(b) Contention isolation")
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_fft_spectral.pdf")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved PSD figure: {out_path}")


def make_latex_table(oversub_feats, contention_feats):
    """Generate LaTeX table of spectral features."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\caption{Spectral features of GPU power traces under baseline and attack conditions. "
                 r"Total energy, entropy, and dominant frequency are computed from Welch PSD estimates "
                 r"(effective $f_s \approx 3.8$\,Hz, Nyquist $\approx 1.9$\,Hz).}")
    lines.append(r"\label{tab:fft_features}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Condition & State & Energy (W$^2$) & Entropy (bits) & Centroid (Hz) & Dom.\ Freq (Hz) \\")
    lines.append(r"\midrule")

    # Oversubscription rows
    lines.append(r"\multicolumn{6}{l}{\textit{Power-budget oversubscription}} \\")
    for _, row in oversub_feats.iterrows():
        lines.append(
            f"  {row['budget']} & {row['condition']} & "
            f"{row['total_energy']:.1f} & {row['spectral_entropy']:.2f} & "
            f"{row['spectral_centroid']:.3f} & {row['dominant_freq_hz']:.3f} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\textit{Contention isolation}} \\")
    for _, row in contention_feats.iterrows():
        label = f"{row['clock_policy'].title()} clock"
        lines.append(
            f"  {label} & {row['condition']} & "
            f"{row['total_energy']:.1f} & {row['spectral_entropy']:.2f} & "
            f"{row['spectral_centroid']:.3f} & {row['dominant_freq_hz']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = os.path.join(FIG_DIR, "tab_fft_features.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table: {out_path}")


def print_summary(oversub_feats, contention_feats):
    """Print key findings to stdout."""
    print("\n=== FFT SPECTRAL ANALYSIS SUMMARY ===\n")

    # Oversubscription: attack vs baseline energy ratio
    print("Power-Budget Oversubscription:")
    for budget in oversub_feats["budget"].unique():
        sub = oversub_feats[oversub_feats["budget"] == budget]
        base_e = sub[sub["condition"] == "Baseline"]["total_energy"].values
        atk_e = sub[sub["condition"] == "Attack"]["total_energy"].values
        if len(base_e) > 0 and len(atk_e) > 0:
            ratio = atk_e[0] / (base_e[0] + 1e-12)
            base_ent = sub[sub["condition"] == "Baseline"]["spectral_entropy"].values[0]
            atk_ent = sub[sub["condition"] == "Attack"]["spectral_entropy"].values[0]
            print(f"  {budget}: energy ratio (atk/base) = {ratio:.1f}x, "
                  f"entropy base={base_ent:.2f} atk={atk_ent:.2f} bits")

    print("\nContention Isolation:")
    for policy in contention_feats["clock_policy"].unique():
        sub = contention_feats[contention_feats["clock_policy"] == policy]
        base_e = sub[sub["condition"] == "Baseline"]["total_energy"].values
        atk_e = sub[sub["condition"] == "Attack"]["total_energy"].values
        if len(base_e) > 0 and len(atk_e) > 0:
            ratio = atk_e[0] / (base_e[0] + 1e-12)
            print(f"  {policy} clock: energy ratio (atk/base) = {ratio:.1f}x")

    # Statistical test: Mann-Whitney U on spectral energy
    print("\nStatistical separation (spectral energy):")
    base_energies = oversub_feats[oversub_feats["condition"] == "Baseline"]["total_energy"].values
    atk_energies = oversub_feats[oversub_feats["condition"] == "Attack"]["total_energy"].values
    if len(base_energies) > 1 and len(atk_energies) > 1:
        u_stat, p_val = stats.mannwhitneyu(base_energies, atk_energies, alternative="two-sided")
        print(f"  Oversubscription: U={u_stat:.0f}, p={p_val:.4f}")


def main():
    print("Computing FFT spectral features from GAIA telemetry...")

    oversub_feats, oversub_curves, budget_labels = analyze_oversubscription_fft()
    contention_feats, contention_curves = analyze_contention_fft()

    make_psd_figure(oversub_curves, budget_labels, contention_curves)
    make_latex_table(oversub_feats, contention_feats)
    print_summary(oversub_feats, contention_feats)

    # Save features CSV for reference
    combined = pd.concat([oversub_feats, contention_feats], ignore_index=True)
    csv_out = os.path.join(FIG_DIR, "fft_spectral_features.csv")
    combined.to_csv(csv_out, index=False)
    print(f"\nSaved features CSV: {csv_out}")
    print("Done.")


if __name__ == "__main__":
    main()
