"""
Project GAIA: Post-Experiment Analysis & Publication Figure Generator
=====================================================================
Run after experiments complete to produce all publication figures,
statistical tables, and metrics required by the proposal.

Usage:
    python analyze_results.py <results_dir>

Where <results_dir> contains:
    t1_3phase_experiment.csv
    t1_sweep_experiment.csv  (optional)
    t3_closed_loop_mitigation.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_csv(results_dir, filename):
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"  [skip] {filename} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [load] {filename}: {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


# =============================================================================
# Figure 1: T1 4-Panel Power Characterization (IEEE format)
# =============================================================================
def figure_t1_power_characterization(df, out_dir, tdp_w=80.0):
    print("\n[Fig 1] T1 Power Characterization (4-panel IEEE)")

    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig, axs = plt.subplots(4, 1, figsize=(7.16, 9), sharex=False)
    plt.subplots_adjust(hspace=0.42)

    phase_colors = {
        'victim_alone': ('#27ae60', 'Baseline (Victim Alone)'),
        'victim_sustained_stress': ('#2980b9', 'Sustained Stress (DC)'),
        'victim_cache_only': ('#8e44ad', 'Cache-Only Stress'),
        'victim_compute_only': ('#f39c12', 'Compute-Only Stress'),
        'victim_pulsed_stress': ('#c0392b', '10 kHz Pulsed Attack'),
        'victim_matched_benign': ('#2980b9', 'Matched Benign Stress'),
    }

    for phase_name, (color, label) in phase_colors.items():
        mask = df['phase'] == phase_name
        if not mask.any():
            continue
        sub = df[mask]
        axs[0].plot(sub['timestamp'], sub['power_w'], color=color, linewidth=0.8, alpha=0.85, label=label)

    axs[0].axhline(y=tdp_w, color='gray', linestyle='--', linewidth=1.0, label=f'TDP ({tdp_w:.0f} W)')
    axs[0].axhline(y=tdp_w * 1.2, color='orange', linestyle=':', linewidth=1.0, label=f'120% Limit ({tdp_w*1.2:.0f} W)')
    axs[0].set_title("(a) GPU Power Draw Across Experimental Conditions", fontweight='bold')
    axs[0].set_ylabel("Power (W)")
    axs[0].legend(fontsize=7, loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    axs[0].grid(True, linestyle=':', alpha=0.5)

    for phase_name, (color, label) in phase_colors.items():
        mask = df['phase'] == phase_name
        if not mask.any():
            continue
        sub = df[mask]
        axs[1].plot(sub['timestamp'], sub['voltage_v'], color=color, linewidth=0.8, alpha=0.85, label=label)

    axs[1].set_title("(b) PDN Voltage Response ($V_{core} = V_{nom} - IR - L \\frac{di}{dt}$)", fontweight='bold')
    axs[1].set_ylabel("Voltage (V)")
    axs[1].legend(fontsize=7, loc='lower right', frameon=True, fancybox=False, edgecolor='gray')
    axs[1].grid(True, linestyle=':', alpha=0.5)

    for phase_name, (color, label) in phase_colors.items():
        mask = df['phase'] == phase_name
        if not mask.any():
            continue
        sub = df[mask]
        axs[2].plot(sub['timestamp'], sub['di_dt_amps_per_sec'], color=color, linewidth=0.7, alpha=0.8, label=label)

    axs[2].set_title("(c) Current Switching Transients ($di/dt$)", fontweight='bold')
    axs[2].set_ylabel("$di/dt$ (A/s)")
    axs[2].legend(fontsize=7, loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    axs[2].grid(True, linestyle=':', alpha=0.5)

    for phase_name, (color, label) in phase_colors.items():
        mask = df['phase'] == phase_name
        if not mask.any():
            continue
        sub = df[mask]
        p_vals = sub['power_w'].values
        if len(p_vals) < 32:
            continue
        dt_mean = np.mean(np.diff(sub['timestamp'].values))
        fs = 1.0 / dt_mean if dt_mean > 0 else 20.0
        nperseg = min(len(p_vals), 256)
        freqs, psd = signal.welch(p_vals, fs=fs, nperseg=nperseg)
        axs[3].semilogy(freqs, psd, color=color, linewidth=1.0, alpha=0.85, label=label)

    axs[3].set_title("(d) Power Spectral Density by Condition", fontweight='bold')
    axs[3].set_xlabel("Frequency (Hz)")
    axs[3].set_ylabel("PSD ($W^2$/Hz)")
    axs[3].legend(fontsize=7, loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    axs[3].grid(True, linestyle=':', alpha=0.5)

    plt.savefig(os.path.join(out_dir, "fig1_t1_power_characterization.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "fig1_t1_power_characterization.pdf"), bbox_inches='tight')
    plt.close()
    print("  [saved] fig1_t1_power_characterization.{png,pdf}")


# =============================================================================
# Figure 2: T1 Victim Latency Degradation (box + strip)
# =============================================================================
def figure_t1_latency_degradation(df, out_dir):
    print("\n[Fig 2] T1 Victim Latency by Condition")

    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

    all_phases = [p for p in df['phase'].unique() if not p.startswith('sweep_')]
    short_labels = {
        'victim_alone': 'Baseline',
        'victim_sustained_stress': 'Sustained',
        'victim_cache_only': 'Cache',
        'victim_compute_only': 'Compute',
        'victim_pulsed_stress': 'Pulsed',
        'victim_matched_benign': 'Benign',
    }
    color_map = {
        'victim_alone': '#27ae60',
        'victim_sustained_stress': '#2980b9',
        'victim_cache_only': '#8e44ad',
        'victim_compute_only': '#f39c12',
        'victim_pulsed_stress': '#c0392b',
        'victim_matched_benign': '#2980b9',
    }
    present = [p for p in all_phases if not df[df['phase'] == p].empty]
    labels = [short_labels.get(p, p.replace('victim_', '')) for p in present]
    colors = [color_map.get(p, '#888888') for p in present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5), gridspec_kw={'width_ratios': [2, 1]})

    data_groups = [df[df['phase'] == pn]['victim_total_latency_ms'].values for pn in present]

    bp = ax1.boxplot(data_groups, labels=labels, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel("Victim Kernel Time (ms)")
    ax1.set_title("(a) Victim Workload Kernel Time Distribution", fontweight='bold')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    trial_means = {}
    for pn in present:
        trial_means[pn] = df[df['phase'] == pn].groupby('trial')['victim_total_latency_ms'].mean().values

    x = np.arange(len(present))
    means = [np.mean(trial_means[pn]) for pn in present]
    stds = [np.std(trial_means[pn], ddof=1) if len(trial_means[pn]) > 1 else 0 for pn in present]
    ax2.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.7, edgecolor='gray')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, rotation=30, ha='right')
    ax2.set_ylabel("Mean Kernel Time (ms)")
    ax2.set_title("(b) Per-Trial Means +/- SD", fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_t1_latency_degradation.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "fig2_t1_latency_degradation.pdf"), bbox_inches='tight')
    plt.close()
    print("  [saved] fig2_t1_latency_degradation.{png,pdf}")


# =============================================================================
# Figure 3: T2 Defense Diagnostics (Kalman + Spectrogram + PCA)
# =============================================================================
def figure_t2_defense_diagnostics(df, out_dir):
    print("\n[Fig 3] T2 Defense Diagnostics (Kalman + Spectrogram + PCA)")

    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig, axs = plt.subplots(3, 1, figsize=(7.16, 9), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
    plt.subplots_adjust(hspace=0.45)

    x_est, p_est = 0.0, 1.0
    Q, R_cov = 1e-3, 1e-1
    kalman_filtered = []
    for val in df['power_w'].values:
        p_pred = p_est + Q
        k_gain = p_pred / (p_pred + R_cov)
        x_est = x_est + k_gain * (val - x_est)
        p_est = (1.0 - k_gain) * p_pred
        kalman_filtered.append(x_est)

    phase_color_map = {
        'victim_alone': '#27ae60',
        'victim_sustained_stress': '#2980b9',
        'victim_cache_only': '#8e44ad',
        'victim_compute_only': '#f39c12',
        'victim_pulsed_stress': '#c0392b',
        'victim_matched_benign': '#2980b9',
    }
    for phase_name, color in phase_color_map.items():
        mask = df['phase'] == phase_name
        if mask.any():
            sub = df[mask]
            axs[0].plot(sub['timestamp'], sub['power_w'], color=color, alpha=0.25, linewidth=0.5)

    axs[0].plot(df['timestamp'], kalman_filtered, color='#2c3e50', linewidth=1.2, label='Kalman Filtered')
    axs[0].set_title("(a) Raw Power Telemetry & Kalman Filter Smoothing", fontweight='bold')
    axs[0].set_ylabel("Power (W)")
    axs[0].legend(fontsize=7)
    axs[0].grid(True, linestyle=':', alpha=0.5)

    dt_mean = np.mean(np.diff(df['timestamp'].values))
    fs_est = 1.0 / dt_mean if dt_mean > 0 else 20.0
    nperseg = min(len(df), 128)
    freqs, times, Sxx = signal.spectrogram(
        df['power_w'].values, fs=fs_est,
        nperseg=nperseg, noverlap=nperseg // 2
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    pcm = axs[1].pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')
    axs[1].set_title("(b) Power Envelope Spectrogram (STFT)", fontweight='bold')
    axs[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(pcm, ax=axs[1], label='Power (dB)')
    axs[1].text(0.01, 0.95, f"$f_s$={fs_est:.1f} Hz | Nyquist={fs_est/2:.1f} Hz",
                transform=axs[1].transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    label_binary = df['phase'].apply(
        lambda x: 1 if 'stress' in str(x).lower() or 'pulsed' in str(x).lower() else 0
    ).values
    window_size = min(len(df) // 10, 50)
    if window_size < 5:
        window_size = 5

    features, labels = [], []
    for i in range(0, len(df) - window_size, window_size):
        sub = df.iloc[i:i+window_size]
        features.append([
            sub['power_w'].mean(),
            sub['power_w'].std(),
            sub['voltage_v'].min(),
            sub['voltage_v'].max() - sub['voltage_v'].min()
        ])
        labels.append(1 if label_binary[i:i+window_size].mean() > 0.5 else 0)

    features = np.array(features)
    labels = np.array(labels)

    if len(features) > 2:
        scaler = StandardScaler()
        projected = PCA(n_components=2).fit_transform(scaler.fit_transform(features))
        benign_mask = labels == 0
        attack_mask = labels == 1
        axs[2].scatter(projected[benign_mask, 0], projected[benign_mask, 1],
                       c='#27ae60', label='Benign', alpha=0.7, edgecolors='none', s=30)
        axs[2].scatter(projected[attack_mask, 0], projected[attack_mask, 1],
                       c='#9b59b6', label='Attack', alpha=0.7, edgecolors='none', s=30)

    axs[2].set_title("(c) PCA Feature Space Projection (Workload State Separability)", fontweight='bold')
    axs[2].set_xlabel("PC1")
    axs[2].set_ylabel("PC2")
    axs[2].legend(fontsize=7)
    axs[2].grid(True, linestyle=':', alpha=0.5)

    plt.savefig(os.path.join(out_dir, "fig3_t2_defense_diagnostics.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "fig3_t2_defense_diagnostics.pdf"), bbox_inches='tight')
    plt.close()
    print("  [saved] fig3_t2_defense_diagnostics.{png,pdf}")


# =============================================================================
# Figure 4: T3 Closed-Loop Mitigation Results
# =============================================================================
def figure_t3_mitigation(df, out_dir):
    print("\n[Fig 4] T3 Closed-Loop Mitigation")

    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.5))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)

    phase_order = ['attack_unmitigated', 'attack_mitigated', 'victim_clamped_baseline', 'recovery_baseline']
    phase_labels = ['Unmitigated\nAttack', 'Mitigated\nAttack', 'Clamped\nBaseline', 'Recovery\nBaseline']
    colors = ['#c0392b', '#e67e22', '#2980b9', '#27ae60']

    # (a) Latency per phase
    data_groups = [df[df['phase'] == pn]['victim_total_latency_ms'].values for pn in phase_order]
    bp = axs[0, 0].boxplot(data_groups, labels=phase_labels, patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axs[0, 0].set_ylabel("Latency (ms)")
    axs[0, 0].set_title("(a) Victim Latency by Phase", fontweight='bold', fontsize=9)
    axs[0, 0].grid(True, axis='y', linestyle=':', alpha=0.5)
    axs[0, 0].tick_params(axis='x', labelsize=7)

    # (b) Power per phase
    data_groups = [df[df['phase'] == pn]['power_w'].values for pn in phase_order]
    bp = axs[0, 1].boxplot(data_groups, labels=phase_labels, patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    axs[0, 1].set_ylabel("Power (W)")
    axs[0, 1].set_title("(b) GPU Power Draw by Phase", fontweight='bold', fontsize=9)
    axs[0, 1].grid(True, axis='y', linestyle=':', alpha=0.5)
    axs[0, 1].tick_params(axis='x', labelsize=7)

    # (c) Mitigation decomposition bar chart
    trial_means = {}
    for pn in phase_order:
        trial_means[pn] = df[df['phase'] == pn].groupby('trial')['victim_total_latency_ms'].mean().values

    if all(pn in trial_means and len(trial_means[pn]) >= 1 for pn in phase_order):
        A = np.mean(trial_means['attack_unmitigated'])
        B = np.mean(trial_means['attack_mitigated'])
        C = np.mean(trial_means['victim_clamped_baseline'])
        D = np.mean(trial_means['recovery_baseline'])
        gross = A - B
        overhead = C - D
        net = gross - overhead

        bars = axs[1, 0].bar(
            ['Gross\nBenefit\n(A−B)', 'Clamping\nOverhead\n(C−D)', 'Net Attack\nReduction'],
            [gross, overhead, net],
            color=['#e67e22', '#2980b9', '#27ae60'],
            edgecolor='gray', alpha=0.7
        )
        axs[1, 0].axhline(y=0, color='gray', linewidth=0.5)
        axs[1, 0].set_ylabel("Latency Difference (ms)")
        axs[1, 0].set_title("(c) Mitigation Decomposition", fontweight='bold', fontsize=9)
        axs[1, 0].grid(True, axis='y', linestyle=':', alpha=0.5)

        for bar, val in zip(bars, [gross, overhead, net]):
            axs[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f'{val:.1f} ms', ha='center', va='bottom', fontsize=7)

    # (d) Time series with mitigation events
    if 'anomaly_detected' in df.columns:
        trial1 = df[df['trial'] == 1] if 'trial' in df.columns else df
        for pn, c in zip(phase_order, colors):
            mask = trial1['phase'] == pn
            if mask.any():
                sub = trial1[mask]
                axs[1, 1].plot(sub['timestamp'], sub['power_w'], color=c, linewidth=0.7, alpha=0.8)

        if 'mitigation_active' in trial1.columns:
            mit_mask = trial1['mitigation_active'] == 1
            if mit_mask.any():
                axs[1, 1].fill_between(
                    trial1['timestamp'], trial1['power_w'].min(), trial1['power_w'].max(),
                    where=mit_mask, alpha=0.15, color='orange', label='Power Clamp Active'
                )

        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Power (W)")
        axs[1, 1].set_title("(d) Power Trace with Mitigation Events", fontweight='bold', fontsize=9)
        axs[1, 1].grid(True, linestyle=':', alpha=0.5)
        axs[1, 1].legend(fontsize=7)

    plt.savefig(os.path.join(out_dir, "fig4_t3_mitigation.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "fig4_t3_mitigation.pdf"), bbox_inches='tight')
    plt.close()
    print("  [saved] fig4_t3_mitigation.{png,pdf}")


# =============================================================================
# Figure 5: Mechanism-Isolation Sweep (if available)
# =============================================================================
def figure_t1_sweep(df, out_dir):
    print("\n[Fig 5] T1 Mechanism-Isolation Sweep")

    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

    sweep_phases = [p for p in df['phase'].unique() if p.startswith('sweep_window_')]
    if not sweep_phases:
        print("  [skip] No sweep phases found")
        return

    core_phases = [p for p in df['phase'].unique() if not p.startswith('sweep_')]
    all_phases = list(core_phases) + sorted(sweep_phases, key=lambda x: int(x.split('_')[-1].replace('ms', '')))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    phase_means = []
    phase_stds = []
    phase_labels = []
    for pn in all_phases:
        trial_vals = df[df['phase'] == pn].groupby('trial')['victim_total_latency_ms'].mean().values
        if len(trial_vals) == 0:
            continue
        phase_means.append(np.mean(trial_vals))
        phase_stds.append(np.std(trial_vals, ddof=1) if len(trial_vals) > 1 else 0.0)
        if pn.startswith('sweep_'):
            phase_labels.append(pn.split('_')[-1])
        else:
            phase_labels.append(pn.replace('victim_', '').replace('_', '\n'))

    x = np.arange(len(phase_means))
    colors_sweep = ['#27ae60' if i == 0 else '#2980b9' if i == 1 else '#c0392b' if i == 2 else '#8e44ad'
                    for i in range(len(phase_means))]
    ax1.bar(x, phase_means, yerr=phase_stds, capsize=3, color=colors_sweep, alpha=0.7, edgecolor='gray')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phase_labels, fontsize=7, rotation=45, ha='right')
    ax1.set_ylabel("Mean Victim Latency (ms)")
    ax1.set_title("(a) Latency vs. Pulse Window", fontweight='bold')
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    power_means = []
    power_stds = []
    for pn in all_phases:
        trial_vals = df[df['phase'] == pn].groupby('trial')['power_w'].mean().values
        if len(trial_vals) == 0:
            continue
        power_means.append(np.mean(trial_vals))
        power_stds.append(np.std(trial_vals, ddof=1) if len(trial_vals) > 1 else 0.0)

    ax2.bar(x[:len(power_means)], power_means, yerr=power_stds, capsize=3,
            color=colors_sweep[:len(power_means)], alpha=0.7, edgecolor='gray')
    ax2.set_xticks(x[:len(power_means)])
    ax2.set_xticklabels(phase_labels[:len(power_means)], fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel("Mean Power (W)")
    ax2.set_title("(b) Power Draw vs. Pulse Window", fontweight='bold')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_t1_sweep.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, "fig5_t1_sweep.pdf"), bbox_inches='tight')
    plt.close()
    print("  [saved] fig5_t1_sweep.{png,pdf}")


# =============================================================================
# Statistical Summary & Metrics Report
# =============================================================================
def generate_metrics_report(t1_df, t3_df, sweep_df, out_dir):
    print("\n[Report] Computing all proposal metrics...")

    lines = []
    lines.append("=" * 72)
    lines.append("PROJECT GAIA: EXPERIMENT RESULTS & METRICS REPORT")
    lines.append("=" * 72)

    # --- T1 Metrics ---
    if t1_df is not None:
        lines.append("\n" + "=" * 72)
        lines.append("TASK 1: ATTACK EFFECTIVENESS METRICS")
        lines.append("=" * 72)

        all_phases = [p for p in t1_df['phase'].unique() if not p.startswith('sweep_')]
        core_phases = [p for p in all_phases if p in t1_df['phase'].values]

        lines.append(f"\n{'Phase':<32}{'Latency(ms)':<18}{'Power(W)':<16}{'Min V(V)':<12}{'Cap Rate%':<10}")
        lines.append("-" * 88)

        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            min_v = sub['voltage_v'].min()
            cap_count = ((sub['sw_power_cap'] == 1) | (sub['hw_slowdown'] == 1)).sum()
            cap_rate = cap_count / len(sub) * 100

            lat_str = f"{trial_lat.mean():.2f} +/- {trial_lat.std():.2f}"
            pow_str = f"{trial_pow.mean():.2f} +/- {trial_pow.std():.2f}"
            lines.append(f"{pn:<32}{lat_str:<18}{pow_str:<16}{min_v:<12.5f}{cap_rate:<10.1f}")

        if 'sm_clock_mhz' in t1_df.columns:
            lines.append("\nClock Frequency by Phase:")
            for pn in core_phases:
                sub = t1_df[t1_df['phase'] == pn]
                clk = sub['sm_clock_mhz']
                if clk.sum() > 0:
                    lines.append(f"  {pn}: {clk.mean():.0f} +/- {clk.std():.0f} MHz")

        if 'gpu_temp_c' in t1_df.columns:
            lines.append("\nGPU Temperature by Phase:")
            for pn in core_phases:
                sub = t1_df[t1_df['phase'] == pn]
                tmp = sub['gpu_temp_c']
                if tmp.sum() > 0:
                    lines.append(f"  {pn}: {tmp.mean():.1f} +/- {tmp.std():.1f} C")

        lines.append("\nPower Variance by Phase:")
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            lines.append(f"  {pn}: var={sub['power_w'].var():.4f} W^2, std={sub['power_w'].std():.4f} W")

        peak_power = t1_df['power_w'].max()
        lines.append(f"\nPeak Power Draw: {peak_power:.2f} W")

        lines.append("\nModule Rail Current di/dt by Phase:")
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            lines.append(f"  {pn} |di/dt|_max: {sub['di_dt_amps_per_sec'].abs().max():.4f} A/s")

        lines.append("\nModule Rail Voltage by Phase:")
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            lines.append(f"  {pn} V_min: {sub['voltage_v'].min():.5f} V")

        lines.append("\nActual Sampling Rate by Phase:")
        for pn in core_phases:
            sub = t1_df[(t1_df['phase'] == pn) & (t1_df['trial'] == 1)]
            dts = sub['timestamp'].diff().dropna()
            if len(dts) > 0:
                lines.append(f"  {pn}: {1/dts.median():.1f} Hz (median dt={dts.median():.3f}s)")

        trial_means = {}
        for pn in core_phases:
            vals = t1_df[t1_df['phase'] == pn].groupby('trial')['victim_total_latency_ms'].mean().values
            if len(vals) > 0:
                trial_means[pn] = vals

        if len(trial_means) >= 2:
            lines.append("\nPaired t-tests (per-trial mean latency):")
            phase_list = list(trial_means.keys())
            for i in range(len(phase_list)):
                for j in range(i+1, len(phase_list)):
                    a, b = phase_list[i], phase_list[j]
                    va, vb = trial_means[a], trial_means[b]
                    if len(va) == len(vb) and len(va) >= 2:
                        t_stat, p_val = ttest_rel(va, vb)
                        d = compute_cohens_d(va, vb)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        lines.append(f"  {a} vs {b}:")
                        lines.append(f"    t={t_stat:.4f}, p={p_val:.6f} {sig}, Cohen's d={d:.3f}")

        lines.append("\nLatency Percentiles:")
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]['victim_total_latency_ms']
            if sub.empty:
                continue
            lines.append(f"  {pn}: p50={sub.quantile(0.50):.2f}, p95={sub.quantile(0.95):.2f}, p99={sub.quantile(0.99):.2f} ms")

    # --- T2 Detection Metrics ---
    if t1_df is not None:
        lines.append("\n" + "=" * 72)
        lines.append("TASK 2: DETECTION & ANOMALY METRICS")
        lines.append("=" * 72)

        label_binary = t1_df['phase'].apply(
            lambda x: 1 if 'stress' in str(x).lower() or 'pulsed' in str(x).lower() else 0
        ).values
        window_size = min(len(t1_df) // 10, 50)
        if window_size < 5:
            window_size = 5

        features, labels = [], []
        for i in range(0, len(t1_df) - window_size, window_size):
            sub = t1_df.iloc[i:i+window_size]
            features.append([
                sub['power_w'].mean(),
                sub['power_w'].std(),
                sub['voltage_v'].min(),
                sub['voltage_v'].max() - sub['voltage_v'].min()
            ])
            labels.append(1 if label_binary[i:i+window_size].mean() > 0.5 else 0)

        features = np.array(features)
        labels = np.array(labels)

        n_benign = (labels == 0).sum()
        n_attack = (labels == 1).sum()
        lines.append(f"\nWindowed samples: {len(labels)} ({n_benign} benign, {n_attack} attack)")
        lines.append(f"Window size: {window_size} telemetry samples")

        if len(features) > 4 and len(np.unique(labels)) == 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            lines.append(f"\nPCA Explained Variance:")
            lines.append(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
            lines.append(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
            lines.append(f"  Total (2 components): {pca.explained_variance_ratio_.sum()*100:.1f}%")

            loo = LeaveOneOut()
            y_true, y_pred = [], []
            for train_idx, test_idx in loo.split(X_scaled):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_scaled[train_idx], labels[train_idx])
                y_true.append(labels[test_idx[0]])
                y_pred.append(clf.predict(X_scaled[test_idx])[0])
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            lines.append(f"\nLogistic Regression (LOO-CV on 4 features):")
            lines.append(f"  Accuracy:  {acc*100:.1f}%")
            lines.append(f"  Precision: {prec*100:.1f}%")
            lines.append(f"  Recall:    {rec*100:.1f}%")
            lines.append(f"  F1-Score:  {f1*100:.1f}%")

            clf_full = LogisticRegression(max_iter=1000).fit(X_scaled, labels)
            lines.append(f"\nFeature weights (full model):")
            feat_names = ['mean_power', 'std_power', 'min_voltage', 'voltage_range']
            for fn, w in zip(feat_names, clf_full.coef_[0]):
                lines.append(f"  {fn}: {w:.4f}")

    # --- T3 Metrics ---
    if t3_df is not None:
        lines.append("\n" + "=" * 72)
        lines.append("TASK 3: MITIGATION EFFECTIVENESS METRICS")
        lines.append("=" * 72)

        phase_order = ['attack_unmitigated', 'attack_mitigated', 'victim_clamped_baseline', 'recovery_baseline']
        lines.append(f"\n{'Phase':<28}{'Latency(ms)':<18}{'Power(W)':<16}{'Min V(V)':<12}")
        lines.append("-" * 74)

        agg = {}
        for pn in phase_order:
            sub = t3_df[t3_df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            min_v = sub['voltage_v'].min()
            agg[pn] = {
                'lat_mean': trial_lat.mean(), 'lat_std': trial_lat.std(),
                'pow_mean': trial_pow.mean(), 'pow_std': trial_pow.std(),
                'min_v': min_v, 'lat_values': trial_lat.values
            }
            lat_str = f"{trial_lat.mean():.2f} ± {trial_lat.std():.2f}"
            pow_str = f"{trial_pow.mean():.2f} ± {trial_pow.std():.2f}"
            lines.append(f"{pn:<28}{lat_str:<18}{pow_str:<16}{min_v:<12.5f}")

        if all(pn in agg for pn in phase_order):
            A = agg['attack_unmitigated']['lat_mean']
            B = agg['attack_mitigated']['lat_mean']
            C = agg['victim_clamped_baseline']['lat_mean']
            D = agg['recovery_baseline']['lat_mean']

            lines.append(f"\nMitigation Decomposition:")
            lines.append(f"  Unmitigated attack latency  (A): {A:.2f} ms")
            lines.append(f"  Mitigated attack latency    (B): {B:.2f} ms")
            lines.append(f"  Clamped baseline latency    (C): {C:.2f} ms")
            lines.append(f"  Recovery baseline latency   (D): {D:.2f} ms")
            lines.append(f"  Gross mitigation benefit  (A-B): {A-B:.2f} ms")
            lines.append(f"  Clamping overhead         (C-D): {C-D:.2f} ms")
            lines.append(f"  Net attack reduction  (A-B)-(C-D): {(A-B)-(C-D):.2f} ms")

            if len(agg['attack_unmitigated']['lat_values']) >= 2:
                t_stat, p_val = ttest_rel(
                    agg['attack_unmitigated']['lat_values'],
                    agg['attack_mitigated']['lat_values']
                )
                d = compute_cohens_d(
                    agg['attack_unmitigated']['lat_values'],
                    agg['attack_mitigated']['lat_values']
                )
                sig = "YES" if p_val < 0.05 else "NO"
                lines.append(f"\n  Paired t-test (unmitigated vs mitigated):")
                lines.append(f"    t={t_stat:.4f}, p={p_val:.6f}, sig={sig}, Cohen's d={d:.3f}")

        # Throttle rate reduction
        for pn in ['attack_unmitigated', 'attack_mitigated']:
            sub = t3_df[t3_df['phase'] == pn]
            if sub.empty:
                continue
            cap_rate = ((sub['sw_power_cap'] == 1) | (sub['hw_slowdown'] == 1)).sum() / len(sub) * 100
            lines.append(f"  {pn} throttle rate: {cap_rate:.1f}%")

    lines.append("\n" + "=" * 72)
    lines.append("END OF METRICS REPORT")
    lines.append("=" * 72)

    report_text = "\n".join(lines)
    report_path = os.path.join(out_dir, "metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  [saved] metrics_report.txt")


# =============================================================================
# LaTeX-ready Tables
# =============================================================================
def generate_latex_tables(t1_df, t3_df, out_dir):
    print("\n[Tables] Generating LaTeX-ready tables...")

    if t1_df is not None:
        core_phases = [p for p in t1_df['phase'].unique() if not p.startswith('sweep_')]
        phase_labels = {
            'victim_alone': 'Baseline (Victim Alone)',
            'victim_sustained_stress': 'Sustained Stress (DC)',
            'victim_cache_only': 'Cache-Only Stress',
            'victim_compute_only': 'Compute-Only Stress',
            'victim_pulsed_stress': '10 kHz Pulsed Attack',
            'victim_matched_benign': 'Matched Benign Stress',
        }

        rows = []
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            min_v = sub['voltage_v'].min()
            cap_count = ((sub['sw_power_cap'] == 1) | (sub['hw_slowdown'] == 1)).sum()
            cap_rate = cap_count / len(sub) * 100
            rows.append({
                'Condition': phase_labels.get(pn, pn),
                'Latency (ms)': f"${trial_lat.mean():.2f} \\pm {trial_lat.std():.2f}$",
                'Power (W)': f"${trial_pow.mean():.2f} \\pm {trial_pow.std():.2f}$",
                '$V_{\\min}$ (V)': f"${min_v:.4f}$",
                'Throttle (\\%)': f"${cap_rate:.1f}$"
            })

        if rows:
            table_df = pd.DataFrame(rows)
            latex = table_df.to_latex(index=False, escape=False, column_format='lcccc')
            path = os.path.join(out_dir, "table_t1_results.tex")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(latex)
            print(f"  [saved] table_t1_results.tex")

    if t3_df is not None:
        phase_order = ['attack_unmitigated', 'attack_mitigated', 'victim_clamped_baseline', 'recovery_baseline']
        phase_labels = {
            'attack_unmitigated': 'Attack (Unmitigated)',
            'attack_mitigated': 'Attack (Mitigated)',
            'victim_clamped_baseline': 'Clamped Baseline',
            'recovery_baseline': 'Recovery Baseline'
        }

        rows = []
        for pn in phase_order:
            sub = t3_df[t3_df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            min_v = sub['voltage_v'].min()
            rows.append({
                'Phase': phase_labels.get(pn, pn),
                'Latency (ms)': f"${trial_lat.mean():.2f} \\pm {trial_lat.std():.2f}$",
                'Power (W)': f"${trial_pow.mean():.2f} \\pm {trial_pow.std():.2f}$",
                '$V_{\\min}$ (V)': f"${min_v:.4f}$"
            })

        if rows:
            table_df = pd.DataFrame(rows)
            latex = table_df.to_latex(index=False, escape=False, column_format='lccc')
            path = os.path.join(out_dir, "table_t3_results.tex")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(latex)
            print(f"  [saved] table_t3_results.tex")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Project GAIA: Post-Experiment Analysis")
    parser.add_argument("results_dir", help="Directory containing experiment CSV files")
    parser.add_argument("--tdp", type=float, default=80.0, help="GPU TDP in watts (for Figure 1)")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("PROJECT GAIA: POST-EXPERIMENT ANALYSIS PIPELINE")
    print(f"Results: {args.results_dir}")
    print(f"Figures: {out_dir}")
    print("=" * 72)

    print("\nLoading datasets...")
    t1_df = load_csv(args.results_dir, "t1_3phase_experiment.csv")
    sweep_df = load_csv(args.results_dir, "t1_sweep_experiment.csv")
    t3_df = load_csv(args.results_dir, "t3_closed_loop_mitigation.csv")

    if t1_df is not None:
        figure_t1_power_characterization(t1_df, out_dir, tdp_w=args.tdp)
        figure_t1_latency_degradation(t1_df, out_dir)
        figure_t2_defense_diagnostics(t1_df, out_dir)

    if sweep_df is not None:
        figure_t1_sweep(sweep_df, out_dir)

    if t3_df is not None:
        figure_t3_mitigation(t3_df, out_dir)

    generate_metrics_report(t1_df, t3_df, sweep_df, out_dir)
    generate_latex_tables(t1_df, t3_df, out_dir)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print(f"All figures and tables in: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
