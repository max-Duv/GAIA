"""Paper figures with explicit trial boundaries and trial-level uncertainty."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'medium',
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .18, 'grid.linewidth': .6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300,
}
CONDITIONS = {
    'victim_alone': ('Baseline', '#555555'),
    'victim_matched_benign': ('Benign control', '#0072B2'),
    'victim_compute_only': ('Compute only', '#E69F00'),
    'victim_cache_only': ('Cache only', '#009E73'),
    'victim_sustained_stress': ('Sustained', '#0072B2'),
    'victim_pulsed_stress': ('Pulsed', '#CC79A7'),
    'victim_temp_sync_stress': ('Temp-sync', '#D55E00'),
}
T3 = {
    'attack_unmitigated': ('Stress / no cap', '#D55E00'),
    'attack_mitigated': ('Stress / response', '#0072B2'),
    'victim_clamped_baseline': ('Baseline / cap', '#E69F00'),
    'recovery_baseline': ('Recovery', '#555555'),
    'attack_terminated': ('Stress / terminated', '#56B4E9'),
}


def kernel_column(df):
    for name in ('victim_kernel_ms', 'victim_gpu_compute_ms', 'victim_total_latency_ms'):
        if name in df:
            return name
    raise ValueError('No victim timing column found')


def trial_segments(df):
    """Never join separate trials or conditions into one signal."""
    for key, group in df.groupby(['trial', 'phase'], sort=False):
        yield key, group.sort_values('timestamp')


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    mean = values.mean()
    half = (student_t.ppf(.975, len(values)-1) * values.std(ddof=1)
            / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return mean, half


def save(fig, out_dir, name, caption):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', bbox_inches='tight', facecolor='white')
    (out / f'{name}_caption.txt').write_text(caption + '\n', encoding='utf-8')
    plt.close(fig)
    print(f'  [saved] {name}.{{png,pdf}} and caption')


def summary(ax, df, column, mapping, xlabel):
    """Each faint dot is one trial; diamonds and intervals summarize trials."""
    present = [p for p in mapping if (df.phase == p).any()]
    for i, phase in enumerate(present):
        values = df[df.phase == phase].groupby('trial')[column].mean().dropna().values
        label, color = mapping[phase]
        ax.scatter(values, i + np.linspace(-.12, .12, len(values)), s=16,
                   color=color, alpha=.5, edgecolors='none', zorder=3)
        mean, ci = mean_ci(values)
        ax.errorbar(mean, i, xerr=ci if np.isfinite(ci) else None,
                    fmt='D', color=color, ms=5, capsize=3, lw=1.5, zorder=4)
    ax.set_yticks(range(len(present)), [mapping[p][0] for p in present])
    ax.set_ylim(len(present)-.5, -.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.margins(x=.15)


@plt.rc_context(STYLE)
def figure_t1_power_characterization(df, out_dir, tdp_w=None):
    phases = [p for p in CONDITIONS if (df.phase == p).any()]
    fig, axes = plt.subplots(2, len(phases), figsize=(7.16, 4.2),
                             sharex=True, sharey='row', squeeze=False,
                             layout='constrained')
    for j, phase in enumerate(phases):
        label, color = CONDITIONS[phase]
        segments = [g for _, g in trial_segments(df[df.phase == phase])]
        for g in segments:
            elapsed = g.timestamp - g.timestamp.iloc[0]
            axes[0, j].plot(elapsed, g.power_w, color=color, alpha=.35, lw=.8)
            axes[1, j].plot(elapsed, g.voltage_v, color=color, alpha=.35, lw=.8)
        axes[0, j].set_title(f'{label}\n(n={len(segments)})', fontsize=9)
        axes[1, j].set_xlabel('Elapsed (s)')
        axes[1, j].ticklabel_format(axis='y', useOffset=False)
        if tdp_w is not None:
            axes[0, j].axhline(tdp_w, color='#333333', ls='--', lw=.8)
    axes[0, 0].set_ylabel('Recorded power (W)')
    axes[1, 0].set_ylabel('Measured rail voltage (V)')
    fig.suptitle('Power and voltage within each condition', fontsize=12)
    save(fig, out_dir, 'fig1_t1_power_characterization',
         'Each line is one trial, aligned to its first sample within a condition. '
         'No lines bridge trial or phase boundaries. Voltage is the logged rail measurement, '
         'not GPU core voltage. Power rail attribution requires the acquisition configuration. '
         + (f'Dashed line: user-specified reference of {tdp_w:g} W; verify rail applicability.'
            if tdp_w is not None else 'No assumed TDP or oversubscription threshold is shown.'))


@plt.rc_context(STYLE)
def figure_t1_latency_degradation(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), layout='constrained')
    summary(axes[0], df, kernel_column(df), CONDITIONS, 'Victim kernel time (ms)')
    summary(axes[1], df, 'power_w', CONDITIONS, 'Recorded power (W)')
    axes[0].set_title('(a) Kernel time')
    axes[1].set_title('(b) Power demand')
    fig.suptitle('Condition comparisons across trials', fontsize=12)
    save(fig, out_dir, 'fig2_t1_latency_degradation',
         'Small dots are individual trial means. Diamonds are equally weighted means of trial '
         'means; error bars are two-sided 95% Student-t confidence intervals across trials '
         '(not independent telemetry samples). Intervals describe repeatability within this '
         'experiment and do not remove systematic measurement bias. Lower kernel time means faster execution.')


@plt.rc_context(STYLE)
def figure_t2_defense_diagnostics(df, out_dir):
    """Describe condition features without presenting PCA as detector validation."""
    features = df.groupby(['trial', 'phase']).agg(
        mean_power=('power_w', 'mean'), power_sd=('power_w', 'std')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6), layout='constrained')
    for phase, (label, color) in CONDITIONS.items():
        g = features[features.phase == phase]
        if len(g):
            axes[0].scatter(g.mean_power, g.power_sd, label=label, color=color,
                            s=28, edgecolors='white', linewidths=.4)
    axes[0].set_xlabel('Trial mean power (W)')
    axes[0].set_ylabel('Within-trial power SD (W)')
    axes[0].set_title('(a) Power features by condition')
    axes[0].legend(loc='best', fontsize=7)
    rows = []
    for (trial, phase), g in trial_segments(df):
        dt = np.diff(g.timestamp)
        dt = dt[dt > 0]
        if len(dt):
            rows.append({'trial': trial, 'phase': phase, 'rate': 1 / np.median(dt)})
    import pandas as pd
    summary(axes[1], pd.DataFrame(rows), 'rate', CONDITIONS, 'Sampling rate (Hz)')
    axes[1].set_title('(b) Achieved sampling rate')
    fig.suptitle('Telemetry characterization', fontsize=12)
    save(fig, out_dir, 'fig3_t2_defense_diagnostics',
         'Left: each point describes one trial-condition segment. Feature separation is '
         'descriptive, not held-out detector accuracy. Right: rates use median positive '
         'timestamp spacing within each segment; points and 95% intervals summarize trials. '
         'Sampling rate is not sensor bandwidth and does not establish kHz waveform resolution.')


@plt.rc_context(STYLE)
def figure_t3_mitigation(df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0), layout='constrained')
    summary(axes[0, 0], df, 'power_w', T3, 'Recorded power (W)')
    summary(axes[0, 1], df, kernel_column(df), T3, 'Victim kernel time (ms)')
    axes[0, 0].set_title('(a) Power across trials')
    axes[0, 1].set_title('(b) Performance cost')
    trial = sorted(df.trial.unique())[0]
    g = df[(df.trial == trial) & (df.phase == 'attack_mitigated')].sort_values('timestamp')
    if not g.empty:
        elapsed = g.timestamp - g.timestamp.iloc[0]
        axes[1, 0].plot(elapsed, g.power_w, color='#0072B2', lw=1.4)
        axes[1, 1].plot(elapsed, g[kernel_column(g)], color='#D55E00', lw=1.4)
        active = g.mitigation_active.to_numpy().astype(bool)
        if active.any():
            onset = elapsed.iloc[np.flatnonzero(active)[0]]
            for ax in axes[1]:
                ax.axvline(onset, color='#333333', ls='--', lw=1)
                ax.axvspan(onset, elapsed.iloc[-1], color='#0072B2', alpha=.08)
                ax.text(.98, .78, f'Activation flag: {onset:.2f} s',
                        transform=ax.transAxes, ha='right', va='top', fontsize=7,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
        for ax in axes[1]:
            ax.set_xlabel('Time since first phase sample (s)')
        axes[1, 0].set_ylabel('Recorded power (W)')
        axes[1, 1].set_ylabel('Victim kernel time (ms)')
    axes[1, 0].set_title(f'(c) Response timeline: trial {trial}')
    axes[1, 1].set_title(f'(d) Kernel timing: trial {trial}')
    fig.suptitle('Power containment and performance tradeoff', fontsize=12)
    save(fig, out_dir, 'fig4_t3_mitigation',
         'Top: trial means (dots), equally weighted aggregate means (diamonds), and 95% '
         'Student-t confidence intervals. Whole-phase means include pre-activation samples. '
         f'Bottom: first trial by identifier ({trial}), selected deterministically, not for effect size. '
         'Dashed line is the logged activation flag; measurements in that row can precede '
         'actuation. Shading marks the period after first activation, not independent proof of cap enforcement.')
    figure_detection(df, out_dir)


@plt.rc_context(STYLE)
def figure_detection(df, out_dir):
    if 'anomaly_detected' not in df:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), layout='constrained')
    rates = df.copy()
    rates['alarm_pct'] = rates.anomaly_detected * 100
    summary(axes[0], rates, 'alarm_pct', T3, 'Samples flagged (%)')
    axes[0].set_xlim(-3, 103)
    axes[0].set_title('(a) Alarm rate, including warm-up')
    for j, phase in enumerate(['attack_unmitigated', 'attack_mitigated']):
        delays, missed = [], 0
        for _, g in trial_segments(df[df.phase == phase]):
            hit = g[g.anomaly_detected == 1]
            if hit.empty:
                missed += 1
            else:
                delays.append(hit.timestamp.iloc[0] - g.timestamp.iloc[0])
        if delays:
            axes[1].scatter(np.repeat(j, len(delays)) + np.linspace(-.06, .06, len(delays)),
                            delays, color=T3[phase][1], s=25)
        axes[1].text(j, .96, f'{len(delays)} detected\n{missed} missed',
                     transform=axes[1].get_xaxis_transform(), ha='center', va='top', fontsize=8)
    axes[1].set_xticks([0, 1], ['No cap', 'Response'])
    axes[1].set_xlim(-.5, 1.5)
    axes[1].set_ylim(0, max(6, axes[1].get_ylim()[1] * 1.3))
    axes[1].set_ylabel('First alarm delay (s)')
    axes[1].set_title('(b) Event detection by trial')
    fig.suptitle('Logged detector behavior', fontsize=12)
    save(fig, out_dir, 'fig6_t2_detection',
         'Alarm rates include every sample, including detector warm-up. Dots and 95% '
         'Student-t intervals summarize per-trial rates. Delays are measured from the first '
         'sample of each stress phase, not exact stress launch. Missed events are counted '
         'explicitly. No-stress baselines do not establish false-positive rates for benign '
         'high-power workloads. Detector identity and threshold must come from the run configuration.')


@plt.rc_context(STYLE)
def figure_t1_sweep(df, out_dir):
    # Support both old window-based and new frequency-based sweep naming
    freq_phases = sorted([p for p in df.phase.unique() if p.startswith('sweep_freq_')],
                         key=lambda p: float(p.replace('sweep_freq_', '').replace('hz', '')))
    window_phases = sorted([p for p in df.phase.unique() if p.startswith('sweep_window_')],
                           key=lambda p: float(p.split('_')[-1][:-2]))
    duty_phases = sorted([p for p in df.phase.unique() if p.startswith('sweep_duty_')],
                         key=lambda p: int(p.replace('sweep_duty_', '').replace('pct', '')))

    if freq_phases:
        x = np.array([float(p.replace('sweep_freq_', '').replace('hz', '')) for p in freq_phases])
        sweep_phases = freq_phases
        x_label = 'Pulsing frequency (Hz)'
    elif window_phases:
        x = np.array([float(p.split('_')[-1][:-2]) for p in window_phases])
        sweep_phases = window_phases
        x_label = 'Configured pulse window (ms)'
    else:
        return

    n_cols = 3 if duty_phases else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7.16 if n_cols == 2 else 10.5, 3.5),
                             layout='constrained')
    for ax, metric, label in zip(axes[:2], ['power_w', kernel_column(df)],
                                 ['Recorded power (W)', 'Victim kernel time (ms)']):
        means, cis = [], []
        for val, phase in zip(x, sweep_phases):
            values = df[df.phase == phase].groupby('trial')[metric].mean().values
            mean, ci = mean_ci(values)
            means.append(mean)
            cis.append(ci)
            ax.scatter(np.repeat(val, len(values)), values, s=14, color='#0072B2', alpha=.4)
        ax.errorbar(x, means, yerr=cis, fmt='o-', color='#0072B2', capsize=3, ms=4)
        ax.set_xscale('log')
        ax.set_xticks(x, [f'{v:g}' for v in x])
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel(x_label)
        ax.set_ylabel(label)
    axes[0].set_title('(a) Power vs frequency')
    axes[1].set_title('(b) Latency vs frequency')

    if duty_phases and n_cols > 2:
        ax = axes[2]
        duty_x = np.array([int(p.replace('sweep_duty_', '').replace('pct', '')) for p in duty_phases])
        for metric, color, label in [('power_w', '#D55E00', 'Power (W)'),
                                     (kernel_column(df), '#0072B2', 'Latency (ms)')]:
            means, cis = [], []
            for d, phase in zip(duty_x, duty_phases):
                values = df[df.phase == phase].groupby('trial')[metric].mean().values
                mean, ci = mean_ci(values)
                means.append(mean)
                cis.append(ci)
            ax.errorbar(duty_x, means, yerr=cis, fmt='o-', color=color, capsize=3, ms=4, label=label)
        ax.set_xlabel('Duty cycle (%)')
        ax.set_xticks(duty_x)
        ax.legend(fontsize=7)
        ax.set_title('(c) Duty-cycle effect')

    fig.suptitle('Attack parameter sensitivity', fontsize=12)
    save(fig, out_dir, 'fig5_t1_sweep',
         'Points summarize equally weighted trial means; error bars are 95% Student-t '
         'confidence intervals. Faint dots are individual trials. Logarithmic x-axis '
         'for frequency sweep. Duty-cycle sweep at fixed 10 kHz. '
         'Connecting lines guide the eye and do not imply measurements between settings.')
