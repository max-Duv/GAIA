import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_power_characterization(df, output_png_path="power_sag_analysis.png", tdp_w=175.0):
    """
    Plots a 4-panel publication-grade IEEE figure analyzing GPU power oversubscription and PDN voltage sags.
    """
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=False)
    plt.subplots_adjust(hspace=0.38)
    
    t = df['timestamp'].values
    p = df['power_w'].values
    v = df['voltage_v'].values
    dI = df['di_dt_amps_per_sec'].values / 1e6 # Convert to A/us

    if 'phase' in df.columns:
        attack_mask = df['phase'].str.contains('pulsed|stress|oversubscribe', case=False, na=False)
        if attack_mask.any():
            attack_t = t[attack_mask.values]
            attack_start, attack_end = attack_t.min(), attack_t.max()
        else:
            attack_start, attack_end = t.min(), t.max()
    elif 'is_attack' in df.columns and df['is_attack'].sum() > 0:
        attack_t = t[df['is_attack'].values.astype(bool)]
        attack_start, attack_end = attack_t.min(), attack_t.max()
    else:
        attack_start, attack_end = 3.0, 8.0

    # Panel 1: Power Draw & Oversubscription Limit
    axs[0].plot(t, p, color='#c0392b', linewidth=1.2, label='Measured GPU Power Draw (W)')
    axs[0].axhline(y=tdp_w, color='#2980b9', linestyle='--', linewidth=1.5, label=f'100% TDP ({tdp_w:.0f}W)')
    axs[0].axhline(y=tdp_w * 1.20, color='#e67e22', linestyle=':', linewidth=1.8, label=f'120% Oversubscription Limit ({tdp_w*1.20:.0f}W)')
    axs[0].axvspan(attack_start, attack_end, color='#f1c40f', alpha=0.18, label='Adversarial Burst Phase')
    axs[0].set_title("(a) Dynamic GPU Power Consumption & 120% Oversubscription Threshold Breach", fontsize=10, fontweight='bold')
    axs[0].set_ylabel("Power (Watts)", fontsize=9)
    axs[0].legend(loc='upper right', fontsize=8, frameon=True)
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Core Voltage Sags (V_core)
    axs[1].plot(t, v, color='#2c3e50', linewidth=1.0, label='Core Supply Voltage ($V_{core}$)')
    axs[1].axhline(y=1.0, color='#7f8c8d', linestyle='--', alpha=0.7, label='Nominal Voltage ($V_{nom} = 1.0V$)')
    axs[1].axhline(y=0.92, color='#c0392b', linestyle=':', linewidth=1.2, label='Safety Margin $V_{min}$ Threshold (0.92V)')
    axs[1].axvspan(attack_start, attack_end, color='#f1c40f', alpha=0.18)
    axs[1].set_title("(b) On-Chip PDN Voltage Transient Sags ($V_{core} = V_{nom} - I\\cdot R - L \\cdot \\frac{di}{dt}$)", fontsize=10, fontweight='bold')
    axs[1].set_ylabel("Voltage (Volts)", fontsize=9)
    axs[1].legend(loc='lower right', fontsize=8, frameon=True)
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Rate of Current Change (di/dt transients)
    axs[2].plot(t, dI, color='#8e44ad', linewidth=0.9, alpha=0.85, label='Current Derivative $\\frac{di}{dt}$')
    axs[2].axvspan(attack_start, attack_end, color='#f1c40f', alpha=0.18)
    axs[2].set_title("(c) Microsecond-Scale Switching Transients (Inductive Kick Generator)", fontsize=10, fontweight='bold')
    axs[2].set_ylabel("$\\frac{di}{dt}$ ($A/\\mu s$)", fontsize=9)
    axs[2].legend(loc='upper right', fontsize=8, frameon=True)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    # Panel 4: Power Spectral Density (PSD / Frequency Analysis)
    dt_mean = np.mean(np.diff(t))
    fs = 1.0 / dt_mean if dt_mean > 0 else 20.0
    nperseg = min(len(p), 1024)
    freqs, psd = signal.welch(p, fs=fs, nperseg=nperseg)
    axs[3].semilogy(freqs, psd, color='#16a085', linewidth=1.2, label='Power Spectral Density (PSD)')
    axs[3].axvline(x=100.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Target Resonant Frequency ($f_0 = 100\\text{ Hz}$)')
    axs[3].set_xlim(0, 300)
    axs[3].set_title("(d) Frequency Domain Characterization (Resonant Mode Identification)", fontsize=10, fontweight='bold')
    axs[3].set_xlabel("Time (Seconds) / Frequency (Hz for Panel d)", fontsize=9)
    axs[3].set_ylabel("$W^2 / \\text{Hz}$", fontsize=9)
    axs[3].legend(loc='upper right', fontsize=8, frameon=True)
    axs[3].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Publication plot successfully generated: {output_png_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python power_analysis_plotter.py <csv_file> [tdp_watts]\n"
            "Requires a T1 experiment CSV with 'power_w' column."
        )

    csv_file = sys.argv[1]
    tdp_w = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    print(f"[+] Ingesting hardware CSV telemetry: {csv_file}")
    df = pd.read_csv(csv_file)

    if 'power_w' not in df.columns:
        raise ValueError(
            f"CSV missing required 'power_w' column. Found columns: {list(df.columns)}"
        )

    out_path = os.path.splitext(csv_file)[0] + "_plot.png"
    plot_power_characterization(df, output_png_path=out_path, tdp_w=tdp_w)
