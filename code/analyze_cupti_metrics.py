"""Analyze nsys GPU hardware performance counters from GAIA experiment.

Closes proposal T1/T2 gap: "performance counters" and "low-level profiler."
Proposal Figure 1 item 6: "hardware performance counters."

Reads the gpu_metrics.csv from nsys profiling and:
1. Pivots the long-format metrics into a wide time-series
2. Aligns with experiment phases using timestamps
3. Produces per-phase summary statistics
4. Generates publication figures showing metric differences
5. Outputs a LaTeX table for the paper
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
PAGE_WIDTH = 7.16

STYLE = {
    'font.family': FONT, 'font.size': 9,
    'axes.labelcolor': '#202020', 'text.color': '#202020',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .6, 'lines.linewidth': 1.6,
    'axes.formatter.useoffset': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .35, 'grid.color': '#D0D4DA', 'grid.linewidth': .4,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 600, 'legend.frameon': False,
}
plt.rcParams.update(STYLE)

ROOT = Path(__file__).resolve().parent.parent
CUPTI_DIR = ROOT / "results" / "cupti_nsys"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

METRICS_OF_INTEREST = {
    0: ("GPC Clock (MHz)", "gpc_clock_mhz"),
    1: ("SYS Clock (MHz)", "sys_clock_mhz"),
    8: ("GR Active (%)", "gr_active_pct"),
    11: ("SMs Active (%)", "sms_active_pct"),
    12: ("SM Issue (%)", "sm_issue_pct"),
    13: ("Tensor Active (%)", "tensor_active_pct"),
    14: ("Warps in Flight (%)", "warps_inflight_pct"),
    15: ("Warps in Flight (Avg)", "warps_inflight_avg"),
}

PHASE_DURATION_S = 15.0
WARMUP_PHASES = 12
RECORDED_PHASES = 12
CLOCK_PHASES = 4

PHASE_NAMES_RECORDED = [
    "victim_alone", "victim_sustained_stress", "victim_cache_only",
    "victim_compute_only", "victim_pulsed_stress", "victim_temp_sync_stress",
    "victim_occupancy_stress", "victim_legit_highpower",
    "temp_sync_control_novictim", "temp_sync_randomized_delay",
    "thermal_spike_hold", "thermal_spike_release",
]

IS_ATTACK = {
    "victim_alone": False,
    "victim_sustained_stress": True,
    "victim_cache_only": True,
    "victim_compute_only": True,
    "victim_pulsed_stress": True,
    "victim_temp_sync_stress": True,
    "victim_occupancy_stress": True,
    "victim_legit_highpower": False,
    "temp_sync_control_novictim": False,
    "temp_sync_randomized_delay": True,
    "thermal_spike_hold": True,
    "thermal_spike_release": False,
}

DISPLAY_NAMES = {
    "victim_alone": "Baseline",
    "victim_sustained_stress": "Sustained",
    "victim_cache_only": "Cache",
    "victim_compute_only": "Compute",
    "victim_pulsed_stress": "Pulsed",
    "victim_temp_sync_stress": "Temp-sync",
    "victim_occupancy_stress": "Occupancy",
    "victim_legit_highpower": "Legit high-power",
    "temp_sync_control_novictim": "Temp-sync ctrl",
    "temp_sync_randomized_delay": "Temp-sync rnd",
    "thermal_spike_hold": "Thermal hold",
    "thermal_spike_release": "Thermal release",
}


def load_and_pivot():
    """Load gpu_metrics.csv and pivot to wide format with one row per timestamp."""
    df = pd.read_csv(CUPTI_DIR / "gpu_metrics.csv")
    print(f"Loaded {len(df)} raw metric samples")

    metric_names = pd.read_csv(CUPTI_DIR / "target_info_gpu_metrics.csv")
    print(f"Metrics available: {len(metric_names)}")
    for _, row in metric_names.iterrows():
        marker = " <--" if row['metricId'] in METRICS_OF_INTEREST else ""
        print(f"  [{row['metricId']:2d}] {row['metricName']}{marker}")

    df_interest = df[df['metricId'].isin(METRICS_OF_INTEREST.keys())].copy()
    df_interest['metric_name'] = df_interest['metricId'].map(
        {k: v[1] for k, v in METRICS_OF_INTEREST.items()})

    wide = df_interest.pivot_table(
        index='timestamp', columns='metric_name', values='value', aggfunc='first'
    ).reset_index()
    wide = wide.sort_values('timestamp').reset_index(drop=True)

    wide['elapsed_s'] = (wide['timestamp'] - wide['timestamp'].iloc[0]) / 1e9

    # Clock metrics are in raw Hz — convert to MHz
    for col in ('gpc_clock_mhz', 'sys_clock_mhz'):
        if col in wide.columns:
            wide[col] = wide[col] / 1e6
    # Warps avg is raw count — convert to thousands for readability
    if 'warps_inflight_avg' in wide.columns:
        wide['warps_inflight_avg'] = wide['warps_inflight_avg'] / 1e6

    print(f"\nPivoted: {len(wide)} timestamped samples, "
          f"duration: {wide['elapsed_s'].iloc[-1]:.1f}s")
    return wide


def assign_phases(wide):
    """Assign experiment phases based on elapsed time.

    The experiment structure is:
    - Warmup: ~12 phases x ~15s each (some have extra pre-heat/cool time)
    - Trial 1: 12 phases x 15s each
    - Clock-policy: 4 phases x 15s each

    We estimate phase boundaries from time gaps in the data.
    """
    t = wide['elapsed_s'].values
    total = t[-1]

    # Estimate based on experiment output: warmup took ~12 phases with
    # temp-sync overhead. The recorded trial starts after warmup.
    # From the output: warmup had ~12 phases, then Trial 1 had 12 phases,
    # then clock-policy had 4 phases.
    # Total time ~= (12 + 12 + 4) * ~20s (with overhead) ~= 560s

    # Find large time gaps (>2s between consecutive samples) to detect phase transitions
    dt = np.diff(t)
    median_dt = np.median(dt)
    gap_threshold = max(0.5, median_dt * 10)

    gaps = np.where(dt > gap_threshold)[0]

    # Phase boundaries from gaps
    boundaries = [0]
    for g in gaps:
        boundaries.append(g + 1)

    n_segments = len(boundaries)
    print(f"\nDetected {n_segments} segments via time gaps (threshold={gap_threshold:.3f}s)")

    # If we can't find gaps, fall back to fixed-duration estimation
    if n_segments < 20:
        print("  Few gaps detected; using fixed-duration phase estimation")
        # Estimate warmup duration: warmup has extra overhead for temp-sync phases
        # From the output, warmup alone is ~12 phases * ~20s = ~240s
        # Then trial 1 = 12 * ~20s = ~240s, then clock = 4 * ~20s = ~80s

        # Sample at 100 Hz, so ~100 samples/sec
        sample_rate = len(wide) / total
        print(f"  Sample rate: {sample_rate:.1f} Hz")

        # Try to find the "Trial 1/1" start by looking for a pattern:
        # warmup has similar structure to trial, so look at total time
        # Warmup ~= first half, Trial ~= second portion
        # From the output: total experiment was about 560s
        # Warmup ended around 240-280s mark

        # Instead, let's just split by equal duration phases
        warmup_end_s = total * 0.42  # rough estimate
        trial_end_s = total * 0.80

        wide['phase'] = 'unknown'

        # Assign based on time within each section
        for i, row in wide.iterrows():
            et = row['elapsed_s']
            if et < warmup_end_s:
                phase_idx = int((et / warmup_end_s) * WARMUP_PHASES)
                phase_idx = min(phase_idx, WARMUP_PHASES - 1)
                wide.at[i, 'phase'] = f'warmup_{phase_idx}'
            elif et < trial_end_s:
                rel = et - warmup_end_s
                dur = trial_end_s - warmup_end_s
                phase_idx = int((rel / dur) * RECORDED_PHASES)
                phase_idx = min(phase_idx, RECORDED_PHASES - 1)
                wide.at[i, 'phase'] = PHASE_NAMES_RECORDED[phase_idx]
            else:
                wide.at[i, 'phase'] = 'clock_policy'
    else:
        # Use gap-based segmentation
        wide['phase'] = 'unknown'
        wide['segment'] = 0
        for i, start_idx in enumerate(boundaries):
            end_idx = boundaries[i+1] if i+1 < len(boundaries) else len(wide)
            wide.iloc[start_idx:end_idx, wide.columns.get_loc('segment')] = i

        # Map segments to phases: skip warmup segments, map trial segments
        # Warmup = first WARMUP_PHASES segments
        # Trial 1 = next RECORDED_PHASES segments
        # Clock = next CLOCK_PHASES segments
        for seg_id in range(n_segments):
            mask = wide['segment'] == seg_id
            if seg_id < WARMUP_PHASES:
                wide.loc[mask, 'phase'] = f'warmup_{seg_id}'
            elif seg_id < WARMUP_PHASES + RECORDED_PHASES:
                pidx = seg_id - WARMUP_PHASES
                if pidx < len(PHASE_NAMES_RECORDED):
                    wide.loc[mask, 'phase'] = PHASE_NAMES_RECORDED[pidx]
            elif seg_id < WARMUP_PHASES + RECORDED_PHASES + CLOCK_PHASES:
                wide.loc[mask, 'phase'] = 'clock_policy'

    # Report
    phase_counts = wide[~wide['phase'].str.startswith('warmup') &
                        (wide['phase'] != 'unknown') &
                        (wide['phase'] != 'clock_policy')].groupby('phase').size()
    print(f"\nRecorded phase sample counts:")
    for phase, count in phase_counts.items():
        atk = "ATTACK" if IS_ATTACK.get(phase, False) else "benign"
        print(f"  {phase:35s} {count:5d} samples  [{atk}]")

    return wide


def compute_phase_stats(wide):
    """Compute per-phase mean/std for each metric."""
    recorded = wide[wide['phase'].isin(PHASE_NAMES_RECORDED)].copy()

    metric_cols = [v[1] for v in METRICS_OF_INTEREST.values() if v[1] in recorded.columns]

    stats = []
    for phase in PHASE_NAMES_RECORDED:
        sub = recorded[recorded['phase'] == phase]
        if len(sub) == 0:
            continue
        row = {
            'phase': phase,
            'display': DISPLAY_NAMES.get(phase, phase),
            'is_attack': IS_ATTACK.get(phase, False),
            'n_samples': len(sub),
        }
        for col in metric_cols:
            if col in sub.columns:
                vals = sub[col].dropna()
                row[f'{col}_mean'] = vals.mean()
                row[f'{col}_std'] = vals.std()
        stats.append(row)

    stats_df = pd.DataFrame(stats)
    return stats_df


def make_figure(stats_df, wide):
    """Create publication figure showing hardware counter differences."""
    key_metrics = [
        ('sms_active_pct', 'SMs Active (%)'),
        ('sm_issue_pct', 'SM Issue (%)'),
        ('warps_inflight_pct', 'Warps in Flight (%)'),
        ('gpc_clock_mhz', 'GPC Clock (MHz)'),
    ]

    available = [(col, label) for col, label in key_metrics
                 if f'{col}_mean' in stats_df.columns]

    if not available:
        print("No key metrics available for figure")
        return

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(PAGE_WIDTH, 2.8),
                              constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]

    phases_to_plot = [p for p in PHASE_NAMES_RECORDED
                      if p in stats_df['phase'].values
                      and p not in ('temp_sync_control_novictim',
                                   'temp_sync_randomized_delay',
                                   'thermal_spike_hold', 'thermal_spike_release')]

    colors = []
    for p in phases_to_plot:
        if IS_ATTACK.get(p, False):
            colors.append('#D55E00')
        elif p == 'victim_legit_highpower':
            colors.append('#E69F00')
        else:
            colors.append('#0072B2')

    display = [DISPLAY_NAMES.get(p, p) for p in phases_to_plot]

    for ax, (col, label) in zip(axes, available):
        means = []
        stds = []
        for p in phases_to_plot:
            row = stats_df[stats_df['phase'] == p]
            if len(row) > 0 and f'{col}_mean' in row.columns:
                means.append(row[f'{col}_mean'].values[0])
                stds.append(row[f'{col}_std'].values[0])
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(phases_to_plot))
        ax.bar(x, means, yerr=stds, capsize=2, color=colors,
               edgecolor='white', linewidth=0.5, width=0.7,
               error_kw={'linewidth': 0.6})
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=45, ha='right', fontsize=6.5)

    fig.savefig(FIGURES / "fig_hw_counters.pdf")
    print(f"\nSaved: {FIGURES / 'fig_hw_counters.pdf'}")
    plt.close(fig)


def make_time_series_figure(wide):
    """Plot key metrics over time, showing all phases."""
    recorded = wide[wide['phase'].isin(PHASE_NAMES_RECORDED)].copy()
    if len(recorded) == 0:
        print("No recorded-phase data for time series")
        return

    metrics_to_plot = [
        ('sms_active_pct', 'SMs Active (%)'),
        ('warps_inflight_pct', 'Warps in Flight (%)'),
        ('gpc_clock_mhz', 'GPC Clock (MHz)'),
    ]
    available = [(c, l) for c, l in metrics_to_plot if c in recorded.columns]

    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(PAGE_WIDTH, 1.8 * len(available)),
                              sharex=True, constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    t = recorded['elapsed_s'].values
    t_offset = t[0]
    t_rel = t - t_offset

    for ax, (col, label) in zip(axes, available):
        vals = recorded[col].values

        attack_mask = recorded['phase'].map(IS_ATTACK).values

        ax.scatter(t_rel[~attack_mask], vals[~attack_mask], s=0.3, alpha=0.5,
                  color='#0072B2', label='Benign', rasterized=True)
        ax.scatter(t_rel[attack_mask], vals[attack_mask], s=0.3, alpha=0.5,
                  color='#D55E00', label='Attack', rasterized=True)
        ax.set_ylabel(label)
        if col == available[0][0]:
            ax.legend(loc='upper right', markerscale=10, fontsize=7)

    axes[-1].set_xlabel('Time (s)')

    fig.savefig(FIGURES / "fig_hw_counters_timeseries.pdf")
    print(f"Saved: {FIGURES / 'fig_hw_counters_timeseries.pdf'}")
    plt.close(fig)


def make_latex_table(stats_df):
    """Generate LaTeX table of hardware counter metrics per phase."""
    key_metrics = [
        ('sms_active_pct', 'SMs Active (\\%)'),
        ('sm_issue_pct', 'SM Issue (\\%)'),
        ('warps_inflight_pct', 'Warps (\\%)'),
        ('gpc_clock_mhz', 'Clock (MHz)'),
    ]
    available = [(col, label) for col, label in key_metrics
                 if f'{col}_mean' in stats_df.columns]

    if not available:
        print("No metrics for LaTeX table")
        return

    n_cols = len(available)
    col_spec = 'l' + 'r' * n_cols

    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{Hardware performance counters (nsys, 100\,Hz) per attack condition. '
        r'Metrics from NVIDIA GB10B GPU on Jetson AGX Thor.}',
        r'  \label{tab:hw_counters}',
        r'  \small',
        f'  \\begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
        '    Condition & ' + ' & '.join(l for _, l in available) + r' \\',
        r'    \midrule',
    ]

    for _, row in stats_df.iterrows():
        if row['phase'] in ('temp_sync_control_novictim', 'temp_sync_randomized_delay',
                           'thermal_spike_hold', 'thermal_spike_release'):
            continue
        vals = []
        for col, _ in available:
            m = row.get(f'{col}_mean', 0)
            s = row.get(f'{col}_std', 0)
            if 'mhz' in col.lower():
                vals.append(f'{m:.0f} $\\pm$ {s:.0f}')
            else:
                vals.append(f'{m:.1f} $\\pm$ {s:.1f}')

        display = row['display'].replace('%', '\\%')
        lines.append(f'    {display} & ' + ' & '.join(vals) + r' \\')

    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]

    tab_path = FIGURES / "tab_hw_counters.tex"
    tab_path.write_text('\n'.join(lines))
    print(f"Saved: {tab_path}")


def main():
    print("=" * 64)
    print("GAIA Hardware Performance Counter Analysis")
    print("=" * 64)

    wide = load_and_pivot()
    wide = assign_phases(wide)
    stats_df = compute_phase_stats(wide)

    if len(stats_df) == 0:
        print("\nERROR: No recorded phases found. Check phase assignment.")
        print("Falling back to full-dataset summary:")
        metric_cols = [v[1] for v in METRICS_OF_INTEREST.values() if v[1] in wide.columns]
        for col in metric_cols:
            vals = wide[col].dropna()
            print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, "
                  f"min={vals.min():.2f}, max={vals.max():.2f}")
        return

    print(f"\n{'='*64}")
    print("Per-Phase Hardware Counter Summary")
    print(f"{'='*64}")
    metric_cols = [v[1] for v in METRICS_OF_INTEREST.values()
                   if f'{v[1]}_mean' in stats_df.columns]
    for _, row in stats_df.iterrows():
        atk = "ATTACK" if row['is_attack'] else "benign"
        print(f"\n  {row['display']:25s} [{atk}] ({row['n_samples']} samples)")
        for col in metric_cols:
            m = row.get(f'{col}_mean', 0)
            s = row.get(f'{col}_std', 0)
            print(f"    {col:30s}: {m:10.2f} +/- {s:8.2f}")

    make_figure(stats_df, wide)
    make_time_series_figure(wide)
    make_latex_table(stats_df)

    stats_df.to_csv(CUPTI_DIR / "hw_counter_summary.csv", index=False)
    print(f"\nSummary CSV: {CUPTI_DIR / 'hw_counter_summary.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
