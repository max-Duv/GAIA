#!/usr/bin/env python3
"""
Project GAIA — H100 DCGM-Based Detection Pipeline (Thrust T2)

Ports the T2 detection pipeline (Kalman filter + threshold + ML classifier)
to run on NVIDIA DCGM telemetry instead of Jetson INA3221 sysfs.

Telemetry source hierarchy:
  1. DCGM (dcgmi dmon) — highest frequency, production-grade
  2. pynvml — available without DCGM daemon
  3. nvidia-smi — lowest frequency fallback

Collects an expanded 8-feature vector per sample:
  [power_w, dI/dt, sm_clock_mhz, gpu_temp_c, gpu_util_pct,
   mem_util_pct, mem_clock_mhz, pcie_tx_bytes]

Runs both real-time monitoring and offline classification against
a pre-trained model or learns one from labeled data.

Usage:
    # Real-time monitoring with threshold detection
    python h100_dcgm_detector.py --mode monitor --duration 60

    # Offline classification on previously collected CSV
    python h100_dcgm_detector.py --mode classify --csv results.csv

    # Collect labeled training data alongside T1 attack experiments
    python h100_dcgm_detector.py --mode collect --duration 30 --label attack
"""

import os
import sys
import time
import csv
import json
import argparse
import subprocess
import threading
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
T2_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "T2")
sys.path.insert(0, T2_DIR)
sys.path.insert(0, SCRIPT_DIR)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, roc_auc_score,
                                  confusion_matrix, classification_report)
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ---------------------------------------------------------------------------
# DCGM telemetry collector
# ---------------------------------------------------------------------------

DCGM_FIELDS = {
    "DCGM_FI_DEV_POWER_USAGE": "power_w",
    "DCGM_FI_DEV_GPU_TEMP": "gpu_temp_c",
    "DCGM_FI_DEV_SM_CLOCK": "sm_clock_mhz",
    "DCGM_FI_DEV_MEM_CLOCK": "mem_clock_mhz",
    "DCGM_FI_DEV_GPU_UTIL": "gpu_util_pct",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "mem_util_pct",
    "DCGM_FI_DEV_PCIE_TX_THROUGHPUT": "pcie_tx_bytes",
}

# DCGM field IDs for dcgmi dmon
DCGM_FIELD_IDS = "150,203,100,101,155,156,205"


class DCGMCollector:
    """Streams telemetry from dcgmi dmon in a background thread."""

    def __init__(self, gpu_idx=0, update_freq_ms=100):
        self.gpu_idx = gpu_idx
        self.update_freq_ms = update_freq_ms
        self._proc = None
        self._thread = None
        self._running = False
        self._latest = {}
        self._lock = threading.Lock()
        self._available = self._check_dcgm()

    def _check_dcgm(self):
        try:
            r = subprocess.run(["dcgmi", "discovery", "-l"],
                               capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, Exception):
            return False

    @property
    def available(self):
        return self._available

    def start(self):
        if not self._available:
            return False

        cmd = [
            "dcgmi", "dmon",
            "-e", DCGM_FIELD_IDS,
            "-d", str(self.update_freq_ms),
            "-i", str(self.gpu_idx),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        time.sleep(0.5)
        return True

    def _reader(self):
        header = None
        for line in self._proc.stdout:
            if not self._running:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("#") and "Entity" in line:
                    header = line.lstrip("#").split()
                continue
            parts = line.split()
            if header and len(parts) >= len(header):
                with self._lock:
                    for i, col in enumerate(header):
                        try:
                            self._latest[col] = float(parts[i])
                        except (ValueError, IndexError):
                            pass

    def read(self):
        with self._lock:
            return dict(self._latest)

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ---------------------------------------------------------------------------
# Unified telemetry reader (DCGM → pynvml → nvidia-smi)
# ---------------------------------------------------------------------------

class H100DetectorTelemetry:
    def __init__(self, gpu_idx=0):
        self.gpu_idx = gpu_idx
        self.source = "none"

        self._dcgm = DCGMCollector(gpu_idx)
        if self._dcgm.available:
            if self._dcgm.start():
                self.source = "dcgm"
                print(f"[telemetry] DCGM streaming active (GPU {gpu_idx})")
                return

        self._pynvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            self.source = "pynvml"
            print(f"[telemetry] Using pynvml (GPU {gpu_idx})")
        except Exception:
            self.source = "nvidia-smi"
            print(f"[telemetry] Falling back to nvidia-smi (GPU {gpu_idx})")

    def read(self):
        if self.source == "dcgm":
            return self._read_dcgm()
        elif self.source == "pynvml":
            return self._read_pynvml()
        return self._read_nvidiasmi()

    def _read_dcgm(self):
        raw = self._dcgm.read()
        return {
            "power_w": raw.get("150", 0.0),
            "gpu_temp_c": raw.get("203", 0.0),
            "sm_clock_mhz": raw.get("100", 0.0),
            "mem_clock_mhz": raw.get("101", 0.0),
            "gpu_util_pct": raw.get("155", 0.0),
            "mem_util_pct": raw.get("156", 0.0),
            "pcie_tx_bytes": raw.get("205", 0.0),
            "provenance": "dcgm",
        }

    def _read_pynvml(self):
        import pynvml
        h = self._pynvml_handle
        result = {"provenance": "pynvml"}
        try:
            result["power_w"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            result["power_w"] = 0.0
        try:
            result["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            result["gpu_temp_c"] = 0.0
        try:
            result["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                h, pynvml.NVML_CLOCK_SM)
        except Exception:
            result["sm_clock_mhz"] = 0.0
        try:
            result["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(
                h, pynvml.NVML_CLOCK_MEM)
        except Exception:
            result["mem_clock_mhz"] = 0.0
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            result["gpu_util_pct"] = u.gpu
            result["mem_util_pct"] = u.memory
        except Exception:
            result["gpu_util_pct"] = 0.0
            result["mem_util_pct"] = 0.0
        try:
            result["pcie_tx_bytes"] = pynvml.nvmlDeviceGetPcieThroughput(
                h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        except Exception:
            result["pcie_tx_bytes"] = 0.0
        return result

    def _read_nvidiasmi(self):
        cmd = [
            "nvidia-smi", f"--id={self.gpu_idx}",
            "--query-gpu=power.draw,temperature.gpu,clocks.current.sm,"
            "clocks.current.memory,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                p = [x.strip() for x in r.stdout.strip().split(",")]
                def _f(v):
                    return float(v) if v not in ("[N/A]", "N/A", "") else 0.0
                return {
                    "power_w": _f(p[0]),
                    "gpu_temp_c": _f(p[1]),
                    "sm_clock_mhz": _f(p[2]),
                    "mem_clock_mhz": _f(p[3]),
                    "gpu_util_pct": _f(p[4]),
                    "mem_util_pct": _f(p[5]),
                    "pcie_tx_bytes": 0.0,
                    "provenance": "nvidia-smi",
                }
        except Exception:
            pass
        return {k: 0.0 for k in [
            "power_w", "gpu_temp_c", "sm_clock_mhz", "mem_clock_mhz",
            "gpu_util_pct", "mem_util_pct", "pcie_tx_bytes", "provenance"
        ]}

    def stop(self):
        if self.source == "dcgm":
            self._dcgm.stop()


# ---------------------------------------------------------------------------
# 8-feature anomaly detector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "power_w", "di_dt", "sm_clock_mhz", "gpu_temp_c",
    "gpu_util_pct", "mem_util_pct", "mem_clock_mhz", "pcie_tx_bytes",
]


class H100AnomalyDetector:
    def __init__(self, threshold_w=200.0, buffer_len=256):
        self.threshold_w = threshold_w
        self.buffer_len = buffer_len
        self.power_buffer = []
        self.time_buffer = []

        # Kalman state
        self.x_est = 0.0
        self.p_est = 1.0
        self.Q = 1e-3
        self.R_cov = 1e-1

        self._prev_power = 0.0
        self._prev_time = time.time()

        self._model = None
        self._scaler = None

    def update(self, t, hw):
        power = hw["power_w"]
        self.time_buffer.append(t)
        self.power_buffer.append(power)
        if len(self.power_buffer) > self.buffer_len:
            self.time_buffer.pop(0)
            self.power_buffer.pop(0)

        # Kalman
        p_pred = self.p_est + self.Q
        k_gain = p_pred / (p_pred + self.R_cov)
        self.x_est = self.x_est + k_gain * (power - self.x_est)
        self.p_est = (1.0 - k_gain) * p_pred

        # dI/dt (using 12V nominal for discrete GPU)
        dt = max(t - self._prev_time, 0.001)
        current = power / 12.0
        prev_current = self._prev_power / 12.0
        di_dt = (current - prev_current) / dt
        self._prev_power = power
        self._prev_time = t

        features = np.array([
            power,
            di_dt,
            hw.get("sm_clock_mhz", 0),
            hw.get("gpu_temp_c", 0),
            hw.get("gpu_util_pct", 0),
            hw.get("mem_util_pct", 0),
            hw.get("mem_clock_mhz", 0),
            hw.get("pcie_tx_bytes", 0),
        ])

        return features

    def detect_threshold(self):
        if len(self.power_buffer) < 16:
            return False, 0.0
        mean_p = np.mean(self.power_buffer)
        return mean_p > self.threshold_w, mean_p

    def detect_ml(self, features):
        if self._model is None or self._scaler is None:
            return None, 0.0
        X = self._scaler.transform(features.reshape(1, -1))
        pred = self._model.predict(X)[0]
        prob = self._model.predict_proba(X)[0][1]
        return bool(pred), float(prob)

    def train(self, X, y, groups=None):
        if not _SKLEARN:
            raise ImportError("scikit-learn required for ML classification")

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if groups is not None and len(np.unique(groups)) > 1:
            logo = LeaveOneGroupOut()
            preds = np.zeros(len(y))
            probs = np.zeros(len(y))
            for train_idx, test_idx in logo.split(X_scaled, y, groups):
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_scaled[train_idx], y[train_idx])
                preds[test_idx] = clf.predict(X_scaled[test_idx])
                probs[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]

            acc = accuracy_score(y, preds)
            prec = precision_score(y, preds, zero_division=0)
            rec = recall_score(y, preds, zero_division=0)
            auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
            cm = confusion_matrix(y, preds)

            print(f"\n  LOTO-CV Results (8-feature RF):")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    ROC-AUC:   {auc:.4f}")
            print(f"    Confusion Matrix:\n{cm}")
        else:
            print("  [!] No group info for LOTO-CV, training on full dataset")

        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X_scaled, y)

        importances = self._model.feature_importances_
        print(f"\n  Feature Importances:")
        for name, imp in sorted(zip(FEATURE_NAMES, importances),
                                 key=lambda x: -x[1]):
            print(f"    {name:20s}: {imp:.4f}")


# ---------------------------------------------------------------------------
# Mode: real-time monitor
# ---------------------------------------------------------------------------

def run_monitor(args):
    tel = H100DetectorTelemetry(gpu_idx=args.gpu)
    det = H100AnomalyDetector(threshold_w=args.threshold, buffer_len=256)

    print(f"\n{'='*70}")
    print(f"  GAIA H100 DCGM DETECTOR — Real-Time Monitor")
    print(f"  Source: {tel.source} | Threshold: {args.threshold}W")
    print(f"  Duration: {args.duration}s")
    print(f"{'='*70}\n")

    header = (f"{'Time':>7s}  {'Power':>8s}  {'Kalman':>8s}  "
              f"{'Temp':>5s}  {'SM%':>4s}  {'Mem%':>4s}  {'Clock':>6s}  {'Status':>12s}")
    print(header)
    print("-" * 70)

    records = []
    start = time.time()

    try:
        while (time.time() - start) < args.duration:
            t = time.time() - start
            hw = tel.read()
            features = det.update(t, hw)
            is_anom, mean_p = det.detect_threshold()

            status = "SAFE"
            if is_anom:
                status = f"ATTACK({mean_p:.0f}W)"

            print(f"{t:7.1f}  {hw['power_w']:8.1f}  {det.x_est:8.1f}  "
                  f"{hw['gpu_temp_c']:5.0f}  {hw['gpu_util_pct']:4.0f}  "
                  f"{hw['mem_util_pct']:4.0f}  {hw['sm_clock_mhz']:6.0f}  "
                  f"{status:>12s}")

            records.append({
                "timestamp": round(t, 4),
                "source": tel.source,
                **{name: round(float(features[i]), 4) for i, name in enumerate(FEATURE_NAMES)},
                "kalman_power": round(det.x_est, 3),
                "threshold_alert": int(is_anom),
                "mean_window_power": round(mean_p, 3),
            })

            time.sleep(max(0.05, 1.0 / args.sample_rate))

    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
    finally:
        tel.stop()

    if records:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(SCRIPT_DIR, f"results_detector_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "dcgm_monitor.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[+] Saved {len(records)} samples to {csv_path}")


# ---------------------------------------------------------------------------
# Mode: offline classify
# ---------------------------------------------------------------------------

def run_classify(args):
    if not _SKLEARN:
        print("[!] scikit-learn required. Install: pip install scikit-learn")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(args.csv)
    print(f"[+] Loaded {len(df)} rows from {args.csv}")

    required = ["power_w", "is_attack"]
    for col in required:
        if col not in df.columns:
            print(f"[!] Missing column: {col}")
            sys.exit(1)

    feature_cols = []
    for name in FEATURE_NAMES:
        if name in df.columns:
            feature_cols.append(name)
        elif name == "di_dt" and "di_dt_amps_per_sec" in df.columns:
            df["di_dt"] = df["di_dt_amps_per_sec"]
            feature_cols.append("di_dt")
        else:
            df[name] = 0.0
            feature_cols.append(name)
            print(f"    [~] Feature '{name}' not in CSV, filling with 0")

    X = df[feature_cols].values.astype(float)
    y = df["is_attack"].values.astype(int)
    groups = df["trial"].values if "trial" in df.columns else None

    print(f"\n  Features: {feature_cols}")
    print(f"  Samples: {len(X)} (attack={y.sum()}, benign={len(y)-y.sum()})")

    det = H100AnomalyDetector()
    det.train(X, y, groups)

    # Save model
    try:
        import joblib
        ts = time.strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(SCRIPT_DIR, f"detector_model_{ts}.joblib")
        joblib.dump({"model": det._model, "scaler": det._scaler,
                     "features": feature_cols}, model_path)
        print(f"\n[+] Model saved: {model_path}")
    except ImportError:
        print("[~] joblib not available, model not saved to disk")


# ---------------------------------------------------------------------------
# Mode: collect labeled data
# ---------------------------------------------------------------------------

def run_collect(args):
    tel = H100DetectorTelemetry(gpu_idx=args.gpu)
    det = H100AnomalyDetector(threshold_w=args.threshold)

    is_attack = 1 if args.label == "attack" else 0
    print(f"\n[+] Collecting labeled data: label={args.label} "
          f"(is_attack={is_attack}), duration={args.duration}s")

    records = []
    start = time.time()

    try:
        while (time.time() - start) < args.duration:
            t = time.time() - start
            hw = tel.read()
            features = det.update(t, hw)

            records.append({
                "timestamp": round(t, 4),
                "is_attack": is_attack,
                "trial": args.trial,
                "source": tel.source,
                **{name: round(float(features[i]), 4) for i, name in enumerate(FEATURE_NAMES)},
            })
            time.sleep(max(0.05, 1.0 / args.sample_rate))

    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
    finally:
        tel.stop()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(SCRIPT_DIR,
                            f"labeled_{args.label}_{ts}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"[+] Saved {len(records)} labeled samples to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAIA H100 DCGM Detector")
    parser.add_argument("--mode", choices=["monitor", "classify", "collect"],
                        default="monitor")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--threshold", type=float, default=200.0,
                        help="Power threshold (W) for anomaly detection")
    parser.add_argument("--sample-rate", type=float, default=10.0,
                        help="Target sampling rate (Hz)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV file for classify mode")
    parser.add_argument("--label", type=str, default="baseline",
                        choices=["baseline", "attack"],
                        help="Label for collect mode")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number for collect mode")
    args = parser.parse_args()

    if args.mode == "monitor":
        run_monitor(args)
    elif args.mode == "classify":
        if not args.csv:
            print("[!] --csv required for classify mode")
            sys.exit(1)
        run_classify(args)
    elif args.mode == "collect":
        run_collect(args)
