"""Paper figures with explicit trial boundaries and trial-level uncertainty."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator, FuncFormatter
from matplotlib import font_manager

# Resolve once; Arial is available on Windows, with a portable bundled fallback.
FONT = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
PAGE_WIDTH = 7.16

STYLE = {
    'font.family': FONT, 'font.size': 9,
    'axes.labelcolor': '#202020', 'text.color': '#202020',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': .7, 'lines.linewidth': 1.6,
    'axes.formatter.useoffset': False,
    'figure.constrained_layout.w_pad': .06,
    'figure.constrained_layout.h_pad': .06,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .6, 'grid.color': '#D9DDE2', 'grid.linewidth': .5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'legend.frameon': False,
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
        fig.savefig(out / f'{name}.{ext}', facecolor='white')
    (out / f'{name}_caption.txt').write_text(caption + '\nLayout: two-column figure, 7.16 inches wide. Insert the PDF at text width; do not shrink to one column.\n', encoding='utf-8')
    plt.close(fig)
    print(f'  [saved] {name}.{{png,pdf}} and caption')


def summary(ax, df, column, mapping, xlabel):
    """Open circles are trials; diamonds and intervals summarize trials."""
    present = [p for p in mapping if (df.phase == p).any()]
    for i, phase in enumerate(present):
        values = df[df.phase == phase].groupby('trial')[column].mean().dropna().values
        label, color = mapping[phase]
        ax.scatter(values, i + (np.linspace(-.15, .15, len(values)) if len(values) > 1 else np.zeros(len(values))), s=20,
                   facecolors='white', edgecolors=color, linewidths=.8, zorder=3)
        mean, ci = mean_ci(values)
        ax.errorbar(mean, i, xerr=ci if np.isfinite(ci) else None,
                    fmt='D', color=color, ms=5, capsize=3, lw=1.6, zorder=4)
    ax.set_yticks(range(len(present)), [mapping[p][0] for p in present])
    ax.set_ylim(len(present)-.5, -.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    ax.margins(x=.12)
    ax.xaxis.set_major_locator(MaxNLocator(4))


@plt.rc_context(STYLE)
def figure_t1_power_characterization(df, out_dir, tdp_w=None):
    """Equal-weight trial summaries in explicit 0.5-second bins, not spaghetti."""
    import pandas as pd
    phases = [p for p in CONDITIONS if (df.phase == p).any()]
    if not phases:
        return
    cols = min(3, len(phases))
    rows = int(np.ceil(len(phases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(PAGE_WIDTH, 1.9 * rows + .25),
                             sharex=True, sharey=True, squeeze=False,
                             layout='constrained')
    for j, phase in enumerate(phases):
        ax = axes.flat[j]
        label, color = CONDITIONS[phase]
        segments = [g for _, g in trial_segments(df[df.phase == phase])]
        binned = []
        for g in segments:
            elapsed = g.timestamp - g.timestamp.iloc[0]
            bins = np.floor(elapsed.to_numpy() / .5).astype(int)
            binned.append(pd.Series(g.power_w.to_numpy()).groupby(bins).mean())
        aligned = pd.concat(binned, axis=1).sort_index()
        aligned = aligned.reindex(range(int(aligned.index.max()) + 1))
        center = aligned.median(axis=1)
        lo, hi = aligned.quantile(.25, axis=1), aligned.quantile(.75, axis=1)
        x = aligned.index.to_numpy() * .5 + .25
        ax.fill_between(x, lo, hi, color=color, alpha=.20, linewidth=0)
        ax.plot(x, center, color=color, lw=1.7)
        ax.set_title(f'({chr(97+j)}) {label}', loc='left')
        ax.text(.97, .08, f'n = {len(segments)}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis='x', visible=False)
        if tdp_w is not None:
            ax.axhline(tdp_w, color='#333333', ls='--', lw=1)
    for ax in list(axes.flat)[len(phases):]:
        ax.set_visible(False)
    fig.supxlabel('Time since first condition sample (s)', fontsize=9)
    fig.supylabel('Recorded power (W)', fontsize=9)
    save(fig, out_dir, 'fig1_t1_power_characterization',
         'Within each trial-condition segment, samples are averaged in nonoverlapping '
         '0.5-second bins aligned to the first sample. Solid lines show the median of '
         'available trial bin means; bands show the interquartile range, not a confidence '
         'interval. Empty bins break the line; end bins may contain fewer trials. '
         'This display summarizes the measured power envelope, not fast electrical transients. '
         + (f'Dashed line: user-specified reference {tdp_w:g} W.' if tdp_w is not None else ''))

    # Voltage is a separate summary so small rail changes do not become crowded traces.
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), layout='constrained')
    summary(axes[0], df, 'voltage_v', CONDITIONS, 'Trial mean rail voltage (V)')
    minima = df.groupby(['trial', 'phase'], as_index=False).voltage_v.min()
    summary(axes[1], minima, 'voltage_v', CONDITIONS, 'Trial minimum rail voltage (V)')
    for i, ax in enumerate(axes):
        ax.set_title(['(a) Mean voltage', '(b) Minimum voltage'][i], loc='left')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'))
    save(fig, out_dir, 'fig7_t1_rail_voltage',
         'Open circles are trial means (left) or trial minima (right). Diamonds and '
         '95% Student-t intervals summarize these trial statistics. Voltage is the '
         'logged rail measurement, not on-die GPU core voltage. Equal values can reflect '
         'sensor quantization; small numerical differences do not establish physical droop.')


@plt.rc_context(STYLE)
def figure_t1_latency_degradation(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), layout='constrained')
    summary(axes[0], df, kernel_column(df), CONDITIONS, 'Victim kernel time (ms)')
    summary(axes[1], df, 'power_w', CONDITIONS, 'Recorded power (W)')
    axes[0].set_title('(a) Kernel time')
    axes[1].set_title('(b) Power demand')
    save(fig, out_dir, 'fig2_t1_latency_degradation',
         'Open circles are individual trial means. Diamonds are equally weighted means of trial '
         'means; error bars are two-sided 95% Student-t confidence intervals across trials '
         '(not independent telemetry samples). Intervals describe repeatability within this '
         'experiment and do not remove systematic measurement bias. Lower kernel time means faster execution.')


@plt.rc_context(STYLE)
def figure_t2_defense_diagnostics(df, out_dir):
    """Describe condition features without presenting PCA as detector validation."""
    features = df.groupby(['trial', 'phase']).agg(
        mean_power=('power_w', 'mean'), power_sd=('power_w', 'std')).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.7), layout='constrained')
    for idx, (phase, (label, color)) in enumerate(CONDITIONS.items()):
        g = features[features.phase == phase]
        if len(g):
            axes[0].scatter(g.mean_power, g.power_sd, label=label, color=color,
                            s=28, marker=['o', 's', '^', 'D', 'v', 'P', 'X'][idx % 7],
                            edgecolors='white', linewidths=.4)
    axes[0].set_xlabel('Trial mean power (W)')
    axes[0].set_ylabel('Within-trial power SD (W)')
    axes[0].set_title('(a) Power features by condition')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.22),
                   ncol=2, fontsize=8, borderaxespad=0, columnspacing=1)
    rows = []
    for (trial, phase), g in trial_segments(df):
        dt = np.diff(g.timestamp)
        dt = dt[dt > 0]
        if len(dt):
            rows.append({'trial': trial, 'phase': phase, 'rate': 1 / np.median(dt)})
    import pandas as pd
    summary(axes[1], pd.DataFrame(rows), 'rate', CONDITIONS, 'Sampling rate (Hz)')
    axes[1].set_title('(b) Achieved sampling rate')
    save(fig, out_dir, 'fig3_t2_defense_diagnostics',
         'Left: each point describes one trial-condition segment. Feature separation is '
         'descriptive, not held-out detector accuracy. Right: rates use median positive '
         'timestamp spacing within each segment; points and 95% intervals summarize trials. '
         'Sampling rate is not sensor bandwidth and does not establish kHz waveform resolution.')


@plt.rc_context(STYLE)
def figure_t3_mitigation(df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_WIDTH, 5.5), layout='constrained')
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
                        transform=ax.transAxes, ha='right', va='top', fontsize=8,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2))
        for ax in axes[1]:
            ax.set_xlabel('Time since first phase sample (s)')
        axes[1, 0].set_ylabel('Recorded power (W)')
        axes[1, 1].set_ylabel('Victim kernel time (ms)')
    axes[1, 0].set_title(f'(c) Response timeline: trial {trial}')
    axes[1, 1].set_title(f'(d) Kernel timing: trial {trial}')
    save(fig, out_dir, 'fig4_t3_mitigation',
         'Top: trial means (open circles), equally weighted aggregate means (diamonds), and 95% '
         'Student-t confidence intervals. Whole-phase means include pre-activation samples. '
         f'Bottom: first trial by identifier ({trial}), selected deterministically, not for effect size. '
         'Dashed line is the logged activation flag; measurements in that row can precede '
         'actuation. Shading marks the period after first activation, not independent proof of cap enforcement.')
    figure_detection(df, out_dir)


@plt.rc_context(STYLE)
def figure_detection(df, out_dir):
    if 'anomaly_detected' not in df:
        return
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), layout='constrained')
    rates = df.copy()
    rates['alarm_pct'] = rates.anomaly_detected * 100
    summary(axes[0], rates, 'alarm_pct', T3, 'Samples flagged (%)')
    axes[0].set_xlim(-3, 103)
    axes[0].set_title('(a) Sample alarm rate')
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

    n_rows = 2 if duty_phases else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(PAGE_WIDTH, 2.65 * n_rows),
                             squeeze=False, layout='constrained')

    def sweep_panel(ax, values_x, phases, metric, ylabel, color, logarithmic=False):
        means, cis = [], []
        for val, phase in zip(values_x, phases):
            values = df[df.phase == phase].groupby('trial')[metric].mean().dropna().values
            mean, ci = mean_ci(values)
            means.append(mean)
            cis.append(ci)
            ax.scatter(np.repeat(val, len(values)), values, s=18,
                       facecolors='white', edgecolors=color, linewidths=.8, zorder=3)
        ax.errorbar(values_x, means, yerr=cis, fmt='D-', color=color,
                    capsize=3, ms=4.5, lw=1.5, zorder=4)
        if logarithmic:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f'{v/1000:g}k' if v >= 1000 else f'{v:g}'))
            ax.xaxis.set_minor_locator(NullLocator())
        else:
            ax.set_xticks(values_x)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel(ylabel)
        ax.grid(axis='x', visible=False)
        ax.margins(x=.08, y=.15)

    for ax, metric, label, title in zip(axes[0], ['power_w', kernel_column(df)],
            ['Recorded power (W)', 'Victim kernel time (ms)'],
            ['(a) Frequency: power', '(b) Frequency: kernel time']):
        sweep_panel(ax, x, sweep_phases, metric, label, '#0072B2', True)
        ax.set_xlabel(x_label.replace('Pulsing', 'Configured pulse'))
        ax.set_title(title if freq_phases else title.replace('Frequency', 'Pulse window'), loc='left')

    if duty_phases:
        duty_x = np.array([int(p.replace('sweep_duty_', '').replace('pct', '')) for p in duty_phases])
        for ax, metric, label, title in zip(axes[1], ['power_w', kernel_column(df)],
                ['Recorded power (W)', 'Victim kernel time (ms)'],
                ['(c) Duty cycle: power', '(d) Duty cycle: kernel time']):
            sweep_panel(ax, duty_x, duty_phases, metric, label, '#D55E00')
            ax.set_xlabel('Configured duty cycle (%)')
            ax.set_title(title, loc='left')

    save(fig, out_dir, 'fig5_t1_sweep',
         'Open circles show individual trial means; diamonds show equally weighted means '
         'of trial means with two-sided 95% Student-t confidence intervals. '
         'The pulse-setting axis is logarithmic. Power and kernel time use separate axes '
         'with explicit units. Connecting lines guide the eye, not fitted models. '
         'Settings are configured values, not independently measured pulse frequencies. '
         + ('Duty-cycle sweep uses the frequency configured in the experiment runner. '
            if duty_phases else ''))


@plt.rc_context(STYLE)
def figure_t3_cnn_mitigation(t3_synth, t3_train, t3_infer, out_dir):
    """Side-by-side mitigation comparison across workload types."""
    datasets = []
    if t3_synth is not None:
        datasets.append(('Synthetic MatMul', t3_synth))
    if t3_train is not None:
        datasets.append(('CNN Training', t3_train))
    if t3_infer is not None:
        datasets.append(('CNN Inference', t3_infer))
    if len(datasets) < 2:
        return

    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(PAGE_WIDTH, 5.0), layout='constrained')
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (wlabel, df) in enumerate(datasets):
        summary(axes[0, col], df, 'power_w', T3, 'Recorded power (W)')
        summary(axes[1, col], df, kernel_column(df), T3, 'Victim latency (ms)')
        axes[0, col].set_title(f'{wlabel}: Power')
        axes[1, col].set_title(f'{wlabel}: Latency')

    save(fig, out_dir, 'fig7_t3_cnn_mitigation',
         'Mitigation comparison across workload types. Open circles are per-trial means; '
         'diamonds with 95% Student-t CIs are equally weighted aggregate means. '
         'Cap mitigation throttles GPU clock (1575→630 MHz); Kill terminates the adversarial process. '
         'Training latency increases under cap because compute-bound kernels slow with lower clock. '
         'Inference latency paradoxically improves under attack because the high clock benefits '
         'lightweight kernels more than contention hurts them.')
