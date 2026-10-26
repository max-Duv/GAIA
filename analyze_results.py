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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "T1"))
from gpu_fault_assessor_v2 import CONDITION_LABELS



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
    figure_t3_cnn_mitigation,
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
                if 'is_attack' in seg.columns:
                    is_attack = int(seg['is_attack'].iloc[0])
                else:
                    lbl = CONDITION_LABELS.get(phase, {})
                    is_attack = int(lbl.get("is_attack", False))
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

    # --- T2 Hard-Negative Evaluation (if legit workloads present) ---
    if t1_df is not None and 'category' in t1_df.columns:
        hard_neg_cats = t1_df[t1_df['category'].isin(
            ['benign_highpower', 'control_tempsync'])]
        if len(hard_neg_cats) > 0:
            lines.append("\n" + "-" * 72)
            lines.append("TASK 2e: HARD-NEGATIVE EVALUATION (legitimate high-power workloads)")
            lines.append("-" * 72)
            for cat in hard_neg_cats['category'].unique():
                sub = t1_df[t1_df['category'] == cat]
                pwr = sub['power_w']
                lines.append(f"  {cat}: n={len(sub)}, "
                             f"mean_power={pwr.mean():.1f}W, "
                             f"std={pwr.std():.1f}W, "
                             f"max={pwr.max():.1f}W")

    # --- T2 Power-Level Detector Metrics (from T3 data) ---
    if t3_df is not None and 'anomaly_detected' in t3_df.columns:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 2b: POWER-LEVEL DETECTOR EVALUATION (from T3 runs)")
        lines.append("-" * 72)

        for pn in t3_df['phase'].unique():
            sub = t3_df[t3_df['phase'] == pn]
            n = len(sub)
            n_det = int(sub['anomaly_detected'].sum())
            is_attack_phase = CONDITION_LABELS.get(pn, {}).get("is_attack", pn.startswith('attack'))
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

        attack_phases = [p for p in t3_df['phase'].unique()
                         if CONDITION_LABELS.get(p, {}).get("is_attack", p.startswith('attack'))]
        benign_phases = [p for p in t3_df['phase'].unique() if p not in attack_phases]
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

        if 'monitoring_overhead_ms' in t3_df.columns:
            overhead = t3_df['monitoring_overhead_ms'].dropna()
            if len(overhead) > 0:
                lines.append(f"\n  Monitoring Overhead:")
                lines.append(f"    mean={overhead.mean():.2f}ms, "
                             f"p50={overhead.quantile(0.5):.2f}ms, "
                             f"p99={overhead.quantile(0.99):.2f}ms, "
                             f"max={overhead.max():.2f}ms")

        if 'adversary_pid' in t3_df.columns:
            pids = t3_df[t3_df['adversary_pid'].notna()]['adversary_pid'].unique()
            if len(pids) > 0:
                lines.append(f"\n  Process Attribution:")
                lines.append(f"    Adversarial PIDs identified: {len(pids)}")
                lines.append(f"    PIDs: {[int(p) for p in pids]}")

    # --- T2 Spectral Characterization (Welch PSD) ---
    if t1_df is not None:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 2c: SPECTRAL CHARACTERIZATION (Welch PSD)")
        lines.append("-" * 72)

        for pn in t1_df['phase'].unique():
            trial_psds = []
            trial_freqs = None
            for trial in sorted(t1_df['trial'].unique()):
                seg = t1_df[(t1_df['trial'] == trial) & (t1_df['phase'] == pn)]
                p_vals = seg['power_w'].values
                if len(p_vals) < 32:
                    continue
                ts = seg['timestamp'].values
                dt_mean = np.mean(np.diff(ts))
                fs = 1.0 / dt_mean if dt_mean > 0 else 4.0
                nperseg = min(len(p_vals), 256)
                freqs, psd = signal.welch(p_vals, fs=fs, nperseg=nperseg)
                if trial_freqs is None or len(freqs) == len(trial_freqs):
                    trial_freqs = freqs
                    trial_psds.append(psd)
            if not trial_psds:
                continue
            avg_psd = np.mean(trial_psds, axis=0)
            fs_rep = 1.0 / dt_mean if dt_mean > 0 else 4.0
            nyquist = fs_rep / 2.0
            peak_idx = np.argmax(avg_psd[1:]) + 1
            peak_freq = trial_freqs[peak_idx]
            peak_psd_db = 10 * np.log10(avg_psd[peak_idx] + 1e-10)
            total_power_db = 10 * np.log10(np.sum(avg_psd) + 1e-10)
            lines.append(f"  {pn:30s}: Fs={fs_rep:.1f}Hz Nyq={nyquist:.1f}Hz | "
                         f"peak={peak_freq:.2f}Hz ({peak_psd_db:.1f}dB) "
                         f"total={total_power_db:.1f}dB [n={len(trial_psds)} trials]")

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

    # --- Temperature-Sync Trigger Analysis ---
    if t1_df is not None and 'temp_sync_trigger' in t1_df.columns:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 1b: TEMPERATURE-SYNC TRIGGER ANALYSIS")
        lines.append("-" * 72)

        ts_data = t1_df[t1_df['phase'] == 'victim_temp_sync_stress']
        if not ts_data.empty:
            for trial in sorted(ts_data['trial'].unique()):
                sub = ts_data[ts_data['trial'] == trial]
                triggers = sub['temp_sync_trigger'].unique()
                trigger_val = [t for t in triggers if t != 'none']
                trigger_str = trigger_val[0] if trigger_val else 'none'
                lines.append(f"  Trial {trial}: trigger={trigger_str}")

            n_temp = len(ts_data[ts_data['temp_sync_trigger'] == 'temp_triggered']['trial'].unique())
            n_timeout = len(ts_data[ts_data['temp_sync_trigger'] == 'timeout_triggered']['trial'].unique())
            total_trials = len(ts_data['trial'].unique())
            lines.append(f"\n  Temperature-triggered: {n_temp}/{total_trials} trials")
            lines.append(f"  Timeout-triggered:     {n_timeout}/{total_trials} trials")

    # --- Termination Recovery Metrics ---
    if t3_df is not None and 'terminated' in t3_df.columns:
        lines.append("\n" + "-" * 72)
        lines.append("TASK 3b: TERMINATION RECOVERY METRICS")
        lines.append("-" * 72)

        term_data = t3_df[t3_df['phase'] == 'attack_terminated']
        if not term_data.empty:
            for trial in sorted(term_data['trial'].unique()):
                sub = term_data[term_data['trial'] == trial]
                pre = sub[sub['terminated'] == 0]
                post = sub[sub['terminated'] == 1]
                if not pre.empty and not post.empty:
                    pre_pow = pre['power_w'].mean()
                    post_pow = post['power_w'].mean()
                    pre_lat = pre['victim_total_latency_ms'].mean()
                    post_lat = post['victim_total_latency_ms'].mean()
                    lines.append(f"  Trial {trial}: power {pre_pow:.1f}W -> {post_pow:.1f}W "
                                 f"(delta {pre_pow - post_pow:.1f}W), "
                                 f"latency {pre_lat:.2f}ms -> {post_lat:.2f}ms")

            all_pre = term_data[term_data['terminated'] == 0]
            all_post = term_data[term_data['terminated'] == 1]
            if not all_pre.empty and not all_post.empty:
                lines.append(f"\n  Aggregate pre-kill:  power={all_pre['power_w'].mean():.1f}W, "
                             f"latency={all_pre['victim_total_latency_ms'].mean():.2f}ms")
                lines.append(f"  Aggregate post-kill: power={all_post['power_w'].mean():.1f}W, "
                             f"latency={all_post['victim_total_latency_ms'].mean():.2f}ms")
                lines.append(f"  Power recovery:  {all_pre['power_w'].mean() - all_post['power_w'].mean():.1f}W")
                lines.append(f"  Latency change:  {all_pre['victim_total_latency_ms'].mean() - all_post['victim_total_latency_ms'].mean():.2f}ms")

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
            'victim_pulsed_stress': '10\\,kHz Pulsed Attack',
            'victim_matched_benign': 'Matched Benign Stress',
            'victim_temp_sync_stress': 'Temp-Sync Pulsed Attack',
            'victim_occupancy_stress': 'Occupancy Stress',
            'victim_legit_highpower': 'Legitimate High-Power',
            'temp_sync_control_novictim': 'Temp-Sync Control (No Victim)',
            'temp_sync_randomized_delay': 'Temp-Sync Randomized Delay',
            'thermal_spike_hold': 'Thermal Spike + Hold',
            'thermal_spike_release': 'Thermal Spike + Release',
        }

        rows = []
        for pn in core_phases:
            sub = t1_df[t1_df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            trial_temp = sub.groupby('trial')['gpu_temp_c'].mean()
            rows.append({
                'Condition': phase_labels.get(pn, pn),
                'Latency (ms)': f"${trial_lat.mean():.2f} \\pm {trial_lat.std():.2f}$",
                'Power (W)': f"${trial_pow.mean():.2f} \\pm {trial_pow.std():.2f}$",
                'Temp ($^\\circ$C)': f"${trial_temp.mean():.1f}$",
            })

        if rows:
            table_df = pd.DataFrame(rows)
            latex = table_df.to_latex(index=False, escape=False, column_format='lccc')
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


def generate_t3_cnn_tables(t3_synth, t3_train, t3_infer, out_dir):
    """Generate cross-workload T3 mitigation comparison table and text report."""
    print("\n[T3-CNN] Generating T3 CNN mitigation comparison tables...")
    from scipy.stats import ttest_ind

    datasets = []
    if t3_synth is not None:
        datasets.append(("Synthetic MatMul", t3_synth))
    if t3_train is not None:
        datasets.append(("CNN Training", t3_train))
    if t3_infer is not None:
        datasets.append(("CNN Inference", t3_infer))
    if not datasets:
        print("  [skip] No T3 data available")
        return

    phase_order = ['victim_clamped_baseline', 'attack_unmitigated', 'attack_mitigated',
                   'attack_terminated', 'recovery_baseline']
    phase_labels = {
        'victim_clamped_baseline': 'Clamped Baseline',
        'attack_unmitigated': 'Unmitigated',
        'attack_mitigated': 'Cap Mitigated',
        'attack_terminated': 'Kill Mitigated',
        'recovery_baseline': 'Recovery',
    }

    lines = ["=" * 72, "T3: CNN VICTIM MITIGATION COMPARISON", "=" * 72]
    tex_rows = []

    for wlabel, df in datasets:
        lines.append(f"\n--- {wlabel} ---")
        agg = {}
        for pn in phase_order:
            sub = df[df['phase'] == pn]
            if sub.empty:
                continue
            trial_lat = sub.groupby('trial')['victim_total_latency_ms'].mean()
            trial_pow = sub.groupby('trial')['power_w'].mean()
            agg[pn] = {
                'lat_mean': trial_lat.mean(), 'lat_std': trial_lat.std(),
                'pow_mean': trial_pow.mean(), 'pow_std': trial_pow.std(),
                'lat_values': trial_lat.values, 'pow_values': trial_pow.values,
            }
            lines.append(f"  {phase_labels.get(pn, pn):<24s}: "
                         f"lat={trial_lat.mean():.2f}±{trial_lat.std():.2f} ms, "
                         f"pwr={trial_pow.mean():.2f}±{trial_pow.std():.2f} W")

        if 'victim_clamped_baseline' in agg and 'attack_unmitigated' in agg:
            base = agg['victim_clamped_baseline']['lat_mean']
            unmit = agg['attack_unmitigated']['lat_mean']
            degrad = (unmit - base) / base * 100
            lines.append(f"  Attack degradation: {degrad:+.1f}%")

            if 'attack_mitigated' in agg:
                cap_lat = agg['attack_mitigated']['lat_mean']
                cap_delta = (cap_lat - base) / base * 100
                lines.append(f"  Cap vs baseline: {cap_delta:+.1f}%")
                t_stat, p_val = ttest_ind(agg['attack_unmitigated']['lat_values'],
                                          agg['attack_mitigated']['lat_values'])
                lines.append(f"  t-test unmit vs cap: t={t_stat:.3f}, p={p_val:.4f}")

            if 'attack_terminated' in agg:
                kill_lat = agg['attack_terminated']['lat_mean']
                kill_delta = (kill_lat - base) / base * 100
                lines.append(f"  Kill vs baseline: {kill_delta:+.1f}%")

            base_pwr = agg['victim_clamped_baseline']['pow_mean']
            unmit_pwr = agg['attack_unmitigated']['pow_mean']
            for strat, key in [('Cap', 'attack_mitigated'), ('Kill', 'attack_terminated')]:
                if key in agg:
                    pwr_red = (unmit_pwr - agg[key]['pow_mean']) / unmit_pwr * 100
                    lines.append(f"  {strat} power reduction: {pwr_red:.1f}%")

        for pn in phase_order:
            if pn not in agg:
                continue
            a = agg[pn]
            lat_str = f"${a['lat_mean']:.2f} \\pm {a['lat_std']:.2f}$"
            pow_str = f"${a['pow_mean']:.2f} \\pm {a['pow_std']:.2f}$"
            tex_rows.append(f"{wlabel} & {phase_labels[pn]} & {lat_str} & {pow_str} \\\\")
        tex_rows.append("\\midrule")

    summary_text = "\n".join(lines)
    txt_path = os.path.join(out_dir, "t3_cnn_mitigation_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  [saved] t3_cnn_mitigation_report.txt")
    print(summary_text)

    if tex_rows and tex_rows[-1] == "\\midrule":
        tex_rows.pop()
    tex = "\\begin{tabular}{llcc}\n\\toprule\n"
    tex += "Workload & Phase & Latency (ms) & Power (W) \\\\\n\\midrule\n"
    tex += "\n".join(tex_rows) + "\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    tex_path = os.path.join(out_dir, "table_t3_cnn_comparison.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"  [saved] table_t3_cnn_comparison.tex")


# =============================================================================
# Detector Breakdown: FPR by benign class, recall by attack type
# =============================================================================
def generate_detector_breakdown(t1_df, out_dir):
    print("\n[Detector] Generating per-condition classifier breakdown...")

    window_size = min(len(t1_df) // 10, 50)
    if window_size < 5:
        window_size = 5

    features, labels, trial_ids, phase_ids = [], [], [], []
    for trial in sorted(t1_df['trial'].unique()):
        for phase in t1_df['phase'].unique():
            seg = t1_df[(t1_df['trial'] == trial) & (t1_df['phase'] == phase)]
            if len(seg) < window_size:
                continue
            if 'is_attack' in seg.columns:
                is_attack = int(seg['is_attack'].iloc[0])
            else:
                lbl = CONDITION_LABELS.get(phase, {})
                is_attack = int(lbl.get("is_attack", False))
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
                phase_ids.append(phase)

    features = np.array(features)
    labels = np.array(labels)
    trial_ids = np.array(trial_ids)
    phase_ids = np.array(phase_ids)

    if len(features) < 5 or len(np.unique(labels)) < 2:
        print("  [skip] Insufficient data for detector breakdown")
        return

    unique_trials = sorted(set(trial_ids))
    y_true_all, y_pred_all, phase_all = [], [], []
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
        y_true_all.extend(labels[test_mask].tolist())
        y_pred_all.extend(preds.tolist())
        phase_all.extend(phase_ids[test_mask].tolist())

    if not y_true_all:
        print("  [skip] No predictions generated")
        return

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    phase_all = np.array(phase_all)

    lines = ["-" * 72,
             "DETECTOR BREAKDOWN: Per-Condition Classifier Performance",
             "-" * 72]

    benign_phases = [p for p in np.unique(phase_all)
                     if not CONDITION_LABELS.get(p, {}).get("is_attack", False)]
    attack_phases = [p for p in np.unique(phase_all)
                     if CONDITION_LABELS.get(p, {}).get("is_attack", False)]

    phase_labels = {
        'victim_alone': 'Baseline (alone)',
        'victim_legit_highpower': 'Legit High-Power ($4096^2$)',
        'temp_sync_control_novictim': 'Temp-Sync Control (no victim)',
        'victim_sustained_stress': 'Sustained Stress',
        'victim_cache_only': 'Cache Eviction',
        'victim_compute_only': 'Compute Saturation',
        'victim_pulsed_stress': 'Pulsed (10 kHz)',
        'victim_temp_sync_stress': 'Temp-Sync Attack',
        'victim_occupancy_stress': 'Occupancy Stress',
        'temp_sync_randomized_delay': 'Temp-Sync Randomized',
        'thermal_spike_hold': 'Thermal Spike + Hold',
        'thermal_spike_release': 'Thermal Spike + Release',
    }

    # --- Benign conditions: FPR ---
    lines.append("\nBenign Conditions (false-positive rate):")
    lines.append(f"  {'Condition':<36s} {'Windows':>7s} {'FP':>4s} {'TN':>4s} {'FPR':>7s}")
    lines.append("  " + "-" * 60)
    tex_benign_rows = []
    for phase in benign_phases:
        mask = phase_all == phase
        pred = y_pred_all[mask]
        fp = int((pred == 1).sum())
        tn = int((pred == 0).sum())
        total = fp + tn
        fpr = fp / max(total, 1)
        name = phase_labels.get(phase, phase)
        lines.append(f"  {name:<36s} {total:>7d} {fp:>4d} {tn:>4d} {fpr:>7.1%}")
        tex_benign_rows.append(
            f"{name} & {total} & {fp} & {tn} & {fpr*100:.1f}\\% \\\\"
        )

    # --- Attack conditions: recall ---
    lines.append("\nAttack Conditions (recall / miss rate):")
    lines.append(f"  {'Condition':<36s} {'Windows':>7s} {'TP':>4s} {'FN':>4s} {'Recall':>7s}")
    lines.append("  " + "-" * 60)
    tex_attack_rows = []
    for phase in attack_phases:
        mask = phase_all == phase
        pred = y_pred_all[mask]
        tp = int((pred == 1).sum())
        fn = int((pred == 0).sum())
        total = tp + fn
        recall = tp / max(total, 1)
        name = phase_labels.get(phase, phase)
        lines.append(f"  {name:<36s} {total:>7d} {tp:>4d} {fn:>4d} {recall:>7.1%}")
        tex_attack_rows.append(
            f"{name} & {total} & {tp} & {fn} & {recall*100:.1f}\\% \\\\"
        )

    # --- Overall confusion ---
    tp_all = int(((y_true_all == 1) & (y_pred_all == 1)).sum())
    fn_all = int(((y_true_all == 1) & (y_pred_all == 0)).sum())
    fp_all = int(((y_true_all == 0) & (y_pred_all == 1)).sum())
    tn_all = int(((y_true_all == 0) & (y_pred_all == 0)).sum())
    lines.append(f"\nOverall Confusion Matrix:")
    lines.append(f"  TP={tp_all}  FP={fp_all}")
    lines.append(f"  FN={fn_all}  TN={tn_all}")
    acc = (tp_all + tn_all) / max(tp_all + fn_all + fp_all + tn_all, 1)
    prec = tp_all / max(tp_all + fp_all, 1)
    rec = tp_all / max(tp_all + fn_all, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    fpr_overall = fp_all / max(fp_all + tn_all, 1)
    lines.append(f"  Accuracy={acc*100:.1f}%  Precision={prec*100:.1f}%  Recall={rec*100:.1f}%  F1={f1*100:.1f}%")
    lines.append(f"  Overall FPR={fpr_overall*100:.1f}%")

    summary = "\n".join(lines)
    txt_path = os.path.join(out_dir, "detector_breakdown.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  [saved] detector_breakdown.txt")
    print(summary)

    # --- LaTeX table ---
    tex = "\\begin{tabular}{lcccc}\n\\toprule\n"
    tex += "\\multicolumn{5}{l}{\\textbf{Benign Conditions (FPR)}} \\\\\n"
    tex += "Condition & Windows & FP & TN & FPR \\\\\n\\midrule\n"
    tex += "\n".join(tex_benign_rows) + "\n"
    tex += "\\midrule\n"
    tex += "\\multicolumn{5}{l}{\\textbf{Attack Conditions (Recall)}} \\\\\n"
    tex += "Condition & Windows & TP & FN & Recall \\\\\n\\midrule\n"
    tex += "\n".join(tex_attack_rows) + "\n"
    tex += "\\bottomrule\n\\end{tabular}\n"

    tex_path = os.path.join(out_dir, "table_detector_breakdown.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"  [saved] table_detector_breakdown.tex")


# =============================================================================
# Workload Comparison Table (synthetic vs training vs inference)
# =============================================================================
def generate_workload_comparison(t1_df, train_df, infer_df, out_dir):
    print("\n[Workload] Generating cross-workload comparison table...")

    workloads = []
    datasets = [
        ("Matrix Multiply ($2048^2$)", "synthetic", t1_df),
        ("CNN Training", "training", train_df),
        ("CNN Inference", "inference", infer_df),
    ]

    for label, wtype, df in datasets:
        if df is None:
            continue
        n_trials = df['trial'].nunique()
        row = {"Workload": label, "Trials": n_trials}
        for phase, col_label in [
            ("victim_alone", "alone"),
            ("victim_sustained_stress", "sustained"),
            ("victim_cache_only", "cache"),
            ("victim_pulsed_stress", "pulsed"),
        ]:
            sub = df[df['phase'] == phase]
            if sub.empty:
                row[f"{col_label}_mean"] = np.nan
                row[f"{col_label}_std"] = np.nan
                continue
            trial_means = sub.groupby('trial')['victim_total_latency_ms'].mean()
            row[f"{col_label}_mean"] = trial_means.mean()
            row[f"{col_label}_std"] = trial_means.std()

        if not np.isnan(row.get("alone_mean", np.nan)) and row["alone_mean"] > 0:
            for atk in ["sustained", "cache", "pulsed"]:
                m = row.get(f"{atk}_mean", np.nan)
                if not np.isnan(m):
                    row[f"{atk}_degrad"] = (m - row["alone_mean"]) / row["alone_mean"] * 100
        workloads.append(row)

    if not workloads:
        print("  [skip] No workload data available")
        return

    # --- Generate LaTeX table ---
    tex_rows = []
    for w in workloads:
        alone_str = f"${w['alone_mean']:.2f} \\pm {w['alone_std']:.2f}$" if not np.isnan(w.get('alone_mean', np.nan)) else "---"
        sust_str = f"${w['sustained_mean']:.2f} \\pm {w['sustained_std']:.2f}$" if not np.isnan(w.get('sustained_mean', np.nan)) else "---"
        cache_str = f"${w['cache_mean']:.2f} \\pm {w['cache_std']:.2f}$" if not np.isnan(w.get('cache_mean', np.nan)) else "---"
        degrad = w.get('sustained_degrad', np.nan)
        degrad_str = f"+{degrad:.0f}\\%" if not np.isnan(degrad) else "---"
        tex_rows.append(
            f"{w['Workload']} & {w['Trials']} & {alone_str} & {sust_str} & {cache_str} & {degrad_str} \\\\"
        )

    tex = "\\begin{tabular}{lccccc}\n\\toprule\n"
    tex += "Workload & Trials & Alone (ms) & Sustained (ms) & Cache (ms) & Degrad. \\\\\n"
    tex += "\\midrule\n"
    tex += "\n".join(tex_rows) + "\n"
    tex += "\\bottomrule\n\\end{tabular}\n"

    path = os.path.join(out_dir, "table_workload_comparison.tex")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f"  [saved] table_workload_comparison.tex")

    # --- Generate text summary ---
    lines = ["\n" + "-" * 72,
             "WORKLOAD-DEPENDENT ATTACK IMPACT",
             "-" * 72]
    for w in workloads:
        lines.append(f"\n  {w['Workload']} ({w['Trials']} trials):")
        for phase_key, phase_label in [("alone", "Alone"), ("sustained", "Sustained"),
                                        ("cache", "Cache"), ("pulsed", "Pulsed")]:
            m = w.get(f"{phase_key}_mean", np.nan)
            s = w.get(f"{phase_key}_std", np.nan)
            if not np.isnan(m):
                lines.append(f"    {phase_label:12s}: {m:.2f} ± {s:.2f} ms")
        for atk in ["sustained", "cache", "pulsed"]:
            d = w.get(f"{atk}_degrad", np.nan)
            if not np.isnan(d):
                lines.append(f"    {atk.capitalize()} degradation: +{d:.1f}%")

    summary = "\n".join(lines)
    summary_path = os.path.join(out_dir, "workload_comparison.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  [saved] workload_comparison.txt")
    print(summary)


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
    t3_train_df = load_csv(args.results_dir, "t3_training_mitigation.csv")
    t3_infer_df = load_csv(args.results_dir, "t3_inference_mitigation.csv")
    train_df = load_csv(args.results_dir, "t1_training_victim.csv")
    infer_df = load_csv(args.results_dir, "t1_inference_victim.csv")

    if t1_df is not None:
        figure_t1_power_characterization(t1_df, out_dir, tdp_w=args.tdp)
        figure_t1_latency_degradation(t1_df, out_dir)
        figure_t2_defense_diagnostics(t1_df, out_dir)

    if sweep_df is not None:
        figure_t1_sweep(sweep_df, out_dir)

    if t3_df is not None:
        figure_t3_mitigation(t3_df, out_dir)

    figure_t3_cnn_mitigation(t3_df, t3_train_df, t3_infer_df, out_dir)

    if not args.figures_only:
        generate_metrics_report(t1_df, t3_df, sweep_df, out_dir)
        generate_latex_tables(t1_df, t3_df, out_dir)
        generate_t3_cnn_tables(t3_df, t3_train_df, t3_infer_df, out_dir)
        generate_workload_comparison(t1_df, train_df, infer_df, out_dir)
        if t1_df is not None:
            generate_detector_breakdown(t1_df, out_dir)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print(f"All figures and tables in: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
