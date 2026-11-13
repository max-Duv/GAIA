"""Cross-workload detector generalization: train on Thrust-1 (matmul/CNN)
telemetry, test on held-out LLM (GPT-2) workloads the detector never saw.

Closes cross-examiner finding #4 (detector evaluated on limited workloads)
and the proposal T2 commitment to detect "unusual workload patterns".

Uses the paper's exact 8-feature set and LogisticRegression detector.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

ROOT = Path(__file__).resolve().parent.parent
T1 = ROOT / "results" / "latest" / "t1_3phase_experiment.csv"
LLM_TRAIN = ROOT / "results" / "jetson_11_12_2026" / "llm_training" / "llm_training.csv"
LLM_INFER = ROOT / "results" / "jetson_11_12_2026" / "llm_inference" / "llm_inference.csv"

FEATS = ['power_w', 'voltage_v', 'di_dt_amps_per_sec',
         'sm_clock_mhz', 'gpu_temp_c', 'gpu_util_pct']
WINDOW = 8  # ~0.25 s at 32 Hz; consistent across train and test


def windowize(df):
    """Build the paper's 8-feature windows, grouped by (trial, phase)."""
    X, y, grp = [], [], []
    for trial in sorted(df['trial'].unique()):
        for phase in df['phase'].unique():
            seg = df[(df['trial'] == trial) & (df['phase'] == phase)]
            if len(seg) < WINDOW:
                continue
            label = int(round(seg['is_attack'].mean()))
            for i in range(0, len(seg) - WINDOW + 1, WINDOW):
                sub = seg.iloc[i:i + WINDOW]
                X.append([
                    sub['power_w'].mean(), sub['power_w'].std(),
                    sub['voltage_v'].min(),
                    sub['voltage_v'].max() - sub['voltage_v'].min(),
                    sub['di_dt_amps_per_sec'].abs().mean(),
                    sub['sm_clock_mhz'].mean(),
                    sub['gpu_temp_c'].mean(),
                    sub['gpu_util_pct'].fillna(0).mean(),
                ])
                y.append(label)
                grp.append(f"{trial}:{phase}")
    return np.nan_to_num(np.array(X)), np.array(y), np.array(grp)


def report(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = ((y_true == 0) & (y_pred == 1)).sum() / max((y_true == 0).sum(), 1)
    print(f"\n  {name}  (n={len(y_true)}, {int((y_true==1).sum())} attack / "
          f"{int((y_true==0).sum())} benign)")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Precision: {prec*100:.1f}%   Recall: {rec*100:.1f}%   F1: {f1*100:.1f}%")
    print(f"    FPR:       {fpr*100:.1f}%")
    if len(np.unique(y_true)) == 2:
        print(f"    ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}   "
              f"PR-AUC: {average_precision_score(y_true, y_prob):.4f}")
        # Best achievable accuracy over all thresholds (separability check)
        order = np.unique(y_prob)
        cands = np.concatenate([[0.0], (order[:-1] + order[1:]) / 2, [1.0]]) if len(order) > 1 else [0.5]
        best_t, best_acc = 0.5, 0.0
        for t in cands:
            a = accuracy_score(y_true, (y_prob >= t).astype(int))
            if a > best_acc:
                best_acc, best_t = a, t
        print(f"    Best-threshold accuracy: {best_acc*100:.1f}% at p>={best_t:.3f} "
              f"(default threshold was 0.5)")
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr)


def main():
    print("=" * 64)
    print("Cross-Workload Detector Generalization (train T1 -> test LLM)")
    print("=" * 64)

    t1 = pd.read_csv(T1)
    lt = pd.read_csv(LLM_TRAIN)
    li = pd.read_csv(LLM_INFER)

    X_tr, y_tr, _ = windowize(t1)
    print(f"\nTrain (Thrust-1 matmul/CNN): {len(X_tr)} windows, "
          f"{int((y_tr==1).sum())} attack / {int((y_tr==0).sum())} benign")

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaler.fit_transform(X_tr), y_tr)

    # Sanity: resubstitution on T1 (not a generalization claim)
    yp = clf.predict(scaler.transform(X_tr))
    print(f"  (T1 in-sample accuracy: {accuracy_score(y_tr, yp)*100:.1f}%)")

    results = {}
    for name, df in [("LLM training (unseen workload)", lt),
                     ("LLM inference (unseen workload)", li)]:
        Xt, yt, _ = windowize(df)
        if len(Xt) == 0:
            print(f"\n  {name}: no windows (segments < {WINDOW})")
            continue
        prob = clf.predict_proba(scaler.transform(Xt))[:, 1]
        pred = clf.predict(scaler.transform(Xt))
        results[name] = report(name, yt, pred, prob)

    # Combined LLM
    Xc, yc = [], []
    for df in (lt, li):
        Xt, yt, _ = windowize(df)
        if len(Xt):
            Xc.append(Xt); yc.append(yt)
    if Xc:
        Xc = np.vstack(Xc); yc = np.concatenate(yc)
        prob = clf.predict_proba(scaler.transform(Xc))[:, 1]
        pred = clf.predict(scaler.transform(Xc))
        results["LLM combined"] = report("LLM combined (unseen)", yc, pred, prob)

    print("\n" + "=" * 64)
    print("Interpretation: the detector was trained ONLY on matmul/CNN")
    print("telemetry and never saw an LLM workload. High accuracy here")
    print("means the power-domain attack signature generalizes across")
    print("workload types; low FPR means benign LLM runs are not flagged.")
    print("=" * 64)


if __name__ == "__main__":
    main()
