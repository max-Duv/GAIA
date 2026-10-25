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
from scipy.stats import ttest_rel
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



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


# Figure rendering is separated from statistical reports for independent review.
from publication_figures import (
    figure_t1_power_characterization, figure_t1_latency_degradation,
    figure_t2_defense_diagnostics, figure_t3_mitigation, figure_t1_sweep,
)


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

        window_size = min(len(t1_df) // 10, 50)
        if window_size < 5:
            window_size = 5

        features, labels, trial_ids = [], [], []
        for trial in sorted(t1_df['trial'].unique()):
            for phase in t1_df['phase'].unique():
                seg = t1_df[(t1_df['trial'] == trial) & (t1_df['phase'] == phase)]
                if len(seg) < window_size:
                    continue
                is_attack = 1 if 'stress' in str(phase).lower() or 'pulsed' in str(phase).lower() else 0
                for i in range(0, len(seg) - window_size + 1, window_size):
                    sub = seg.iloc[i:i+window_size]
                    features.append([
                        sub['power_w'].mean(),
                        sub['power_w'].std(),
                        sub['voltage_v'].min(),
                        sub['voltage_v'].max() - sub['voltage_v'].min()
                    ])
                    labels.append(is_attack)
                    trial_ids.append(int(trial))

        features = np.array(features)
        labels = np.array(labels)
        trial_ids = np.array(trial_ids)

        n_benign = (labels == 0).sum()
        n_attack = (labels == 1).sum()
        lines.append(f"\nWindowed samples: {len(labels)} ({n_benign} benign, {n_attack} attack)")
        lines.append(f"Window size: {window_size} telemetry samples")

        if len(features) > 4 and len(np.unique(labels)) == 2:
            scaler_full = StandardScaler()
            X_full = scaler_full.fit_transform(features)
            pca_full = PCA(n_components=2)
            pca_full.fit_transform(X_full)

            lines.append(f"\nPCA Explained Variance:")
            lines.append(f"  PC1: {pca_full.explained_variance_ratio_[0]*100:.1f}%")
            lines.append(f"  PC2: {pca_full.explained_variance_ratio_[1]*100:.1f}%")
            lines.append(f"  Total (2 components): {pca_full.explained_variance_ratio_.sum()*100:.1f}%")

            unique_trials = sorted(set(trial_ids))
            y_true_lot, y_pred_lot = [], []
            for held_out in unique_trials:
                train_mask = trial_ids != held_out
                test_mask = trial_ids == held_out
                if labels[test_mask].size == 0 or labels[train_mask].size == 0:
                    continue
                if len(np.unique(labels[train_mask])) < 2:
                    continue
                s = StandardScaler()
                X_train = s.fit_transform(features[train_mask])
                X_test = s.transform(features[test_mask])
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, labels[train_mask])
                preds = clf.predict(X_test)
                y_true_lot.extend(labels[test_mask].tolist())
                y_pred_lot.extend(preds.tolist())

            if len(y_true_lot) > 0:
                y_true_lot = np.array(y_true_lot)
                y_pred_lot = np.array(y_pred_lot)
                acc = accuracy_score(y_true_lot, y_pred_lot)
                prec = precision_score(y_true_lot, y_pred_lot, zero_division=0)
                rec = recall_score(y_true_lot, y_pred_lot, zero_division=0)
                f1 = f1_score(y_true_lot, y_pred_lot, zero_division=0)

                lines.append(f"\nLogistic Regression (leave-one-trial-out CV, scaler fit per fold):")
                lines.append(f"  Trials held out: {len(unique_trials)}")
                lines.append(f"  Accuracy:  {acc*100:.1f}%")
                lines.append(f"  Precision: {prec*100:.1f}%")
                lines.append(f"  Recall:    {rec*100:.1f}%")
                lines.append(f"  F1-Score:  {f1*100:.1f}%")

    # --- T2 Power-Level Detector Metrics (from T3 data) ---
    if t3_df is not None and 'anomaly_detected' in t3_df.columns:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 2b: POWER-LEVEL DETECTOR EVALUATION (from T3 runs)")
        lines.append("-" * 72)

        for pn in t3_df['phase'].unique():
            sub = t3_df[t3_df['phase'] == pn]
            n = len(sub)
            n_det = int(sub['anomaly_detected'].sum())
            is_attack_phase = pn.startswith('attack')
            ground_truth = 1 if is_attack_phase else 0
            if ground_truth == 1:
                tp = n_det
                fn = n - n_det
                tpr = tp / max(tp + fn, 1)
                lines.append(f"  {pn:30s}: {n_det:3d}/{n} detected (TPR={tpr:.1%})")
            else:
                fp = n_det
                tn = n - n_det
                fpr = fp / max(fp + tn, 1)
                lines.append(f"  {pn:30s}: {n_det:3d}/{n} detected (FPR={fpr:.1%})")

        attack_phases = [p for p in t3_df['phase'].unique() if p.startswith('attack')]
        benign_phases = [p for p in t3_df['phase'].unique() if not p.startswith('attack')]
        tp_total = int(t3_df[t3_df['phase'].isin(attack_phases)]['anomaly_detected'].sum())
        fn_total = len(t3_df[t3_df['phase'].isin(attack_phases)]) - tp_total
        fp_total = int(t3_df[t3_df['phase'].isin(benign_phases)]['anomaly_detected'].sum())
        tn_total = len(t3_df[t3_df['phase'].isin(benign_phases)]) - fp_total

        spec_prec = tp_total / max(tp_total + fp_total, 1)
        spec_rec = tp_total / max(tp_total + fn_total, 1)
        spec_f1 = 2 * spec_prec * spec_rec / max(spec_prec + spec_rec, 1e-10)
        spec_acc = (tp_total + tn_total) / max(tp_total + fn_total + fp_total + tn_total, 1)

        lines.append(f"\n  Power-Level Detector Aggregate:")
        lines.append(f"    TP={tp_total}, FN={fn_total}, FP={fp_total}, TN={tn_total}")
        lines.append(f"    Accuracy:  {spec_acc*100:.1f}%")
        lines.append(f"    Precision: {spec_prec*100:.1f}%")
        lines.append(f"    Recall:    {spec_rec*100:.1f}%")
        lines.append(f"    F1-Score:  {spec_f1*100:.1f}%")

        first_det_times = []
        for trial in sorted(t3_df['trial'].unique()):
            for pn in attack_phases:
                td = t3_df[(t3_df['trial'] == trial) & (t3_df['phase'] == pn)]
                det_rows = td[td['anomaly_detected'] == 1]
                if len(det_rows) > 0:
                    phase_start = td['timestamp'].min()
                    first_det = det_rows['timestamp'].min()
                    first_det_times.append(first_det - phase_start)
        if first_det_times:
            lines.append(f"\n  Detection Latency (time from phase start to first detection):")
            lines.append(f"    Mean: {np.mean(first_det_times):.1f}s")
            lines.append(f"    Min:  {np.min(first_det_times):.1f}s")
            lines.append(f"    Max:  {np.max(first_det_times):.1f}s")
        else:
            lines.append(f"\n  Detection Latency: No detections during attack phases")

    # --- T3 Metrics ---
    if t3_df is not None:
        lines.append("\n" + "=" * 72)
        lines.append("TASK 3: MITIGATION EFFECTIVENESS METRICS")
        lines.append("=" * 72)

        phase_order = ['attack_unmitigated', 'attack_mitigated', 'attack_terminated', 'victim_clamped_baseline', 'recovery_baseline']
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

    # --- T2 Spectral Characterization (Welch PSD) ---
    if t1_df is not None:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 2c: SPECTRAL CHARACTERIZATION (Welch PSD)")
        lines.append("-" * 72)

        for pn in t1_df['phase'].unique():
            sub = t1_df[t1_df['phase'] == pn]
            p_vals = sub['power_w'].values
            if len(p_vals) < 32:
                continue
            ts = sub['timestamp'].values
            dt_mean = np.mean(np.diff(ts))
            fs = 1.0 / dt_mean if dt_mean > 0 else 4.0
            nyquist = fs / 2.0
            nperseg = min(len(p_vals), 256)
            freqs, psd = signal.welch(p_vals, fs=fs, nperseg=nperseg)
            peak_idx = np.argmax(psd[1:]) + 1
            peak_freq = freqs[peak_idx]
            peak_psd_db = 10 * np.log10(psd[peak_idx] + 1e-10)
            total_power_db = 10 * np.log10(np.sum(psd) + 1e-10)
            lines.append(f"  {pn:30s}: Fs={fs:.1f}Hz Nyquist={nyquist:.1f}Hz | "
                         f"peak={peak_freq:.2f}Hz ({peak_psd_db:.1f}dB) total={total_power_db:.1f}dB")

        lines.append(f"\n  Note: Attack pulsing at 10kHz is far above the ~2Hz Nyquist limit.")
        lines.append(f"  Spectral analysis characterizes the low-frequency power envelope,")
        lines.append(f"  not the kHz-scale attack modulation pattern.")

    # --- GPU Utilization Metrics (Performance Counters) ---
    if t1_df is not None and 'gpu_util_pct' in t1_df.columns:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 2d: GPU PERFORMANCE COUNTER METRICS")
        lines.append("-" * 72)

        for pn in t1_df['phase'].unique():
            sub = t1_df[t1_df['phase'] == pn]
            gpu_u = sub['gpu_util_pct']
            mem_u = sub['mem_util_pct']
            if gpu_u.max() > 0 or mem_u.max() > 0:
                lines.append(f"  {pn:30s}: GPU={gpu_u.mean():.0f}% Mem={mem_u.mean():.0f}%")
            else:
                lines.append(f"  {pn:30s}: GPU=N/A Mem=N/A (counters returned 0)")

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
            'victim_temp_sync_stress': 'Temp-Sync Pulsed Attack',
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
        phase_order = ['attack_unmitigated', 'attack_mitigated', 'attack_terminated', 'victim_clamped_baseline', 'recovery_baseline']
        phase_labels = {
            'attack_unmitigated': 'Attack (Unmitigated)',
            'attack_mitigated': 'Attack (Mitigated)',
            'attack_terminated': 'Attack (Terminated)',
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
    parser.add_argument("--tdp", type=float, default=None,
                        help="Optional verified power reference in watts; no reference by default")
    parser.add_argument("--out-dir", help="Output folder (default: results_dir/figures)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Render figures without regenerating statistical reports or tables")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "figures")
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

    if not args.figures_only:
        generate_metrics_report(t1_df, t3_df, sweep_df, out_dir)
        generate_latex_tables(t1_df, t3_df, out_dir)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print(f"All figures and tables in: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
