import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class PlatformAgnosticAnomalyDetector:
    """
    Project GAIA: Thrust T2 Signal Processing & Defense Filters (Version 2)

    Implements:
    - Real-time Kalman filtering for telemetry noise smoothing.
    - Sliding-window Short-Time Fourier Transform (STFT) for resonant attack isolation.
    - Multi-feature Principal Component Analysis (PCA) for workload state separability.
    """
    def __init__(self, sampling_rate_hz=20.0, buffer_len=256, gpu_idx=0):
        self.sampling_rate_hz = sampling_rate_hz
        self.buffer_len = buffer_len
        self.gpu_idx = gpu_idx
        self.power_buffer = []
        self.voltage_buffer = []
        self.time_buffer = []

        self.x_est = 0.0
        self.p_est = 1.0
        self.Q = 1e-3
        self.R_cov = 1e-1

        self.jetson_paths = self._detect_jetson_ina3221()
        self._nvml_handle = None
        self._has_pynvml = False
        if not self.jetson_paths:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_idx)
                self._has_pynvml = True
            except Exception:
                self._nvml_handle = None
                self._has_pynvml = False

        if self.jetson_paths:
            self.platform_name = "NVIDIA Jetson Orin (sysfs INA3221)"
        elif self._has_pynvml:
            self.platform_name = "Discrete NVIDIA GPU (pynvml)"
        else:
            raise RuntimeError(
                "No hardware telemetry source detected. "
                "Requires either Jetson INA3221 sysfs nodes or pynvml (pip install nvidia-ml-py). "
                "Verify GPU is present and drivers are loaded."
            )

    def _detect_jetson_ina3221(self):
        hwmon_base = "/sys/class/hwmon"
        paths = {}
        if not os.path.exists(hwmon_base):
            return None

        try:
            for hwmon_dir in os.listdir(hwmon_base):
                full_dir = os.path.join(hwmon_base, hwmon_dir)
                name_file = os.path.join(full_dir, "name")
                if os.path.exists(name_file):
                    with open(name_file, "r") as f:
                        name = f.read().strip()
                    if "ina3221" in name or "tegra_mgb" in name:
                        for i in range(1, 4):
                            p_file = os.path.join(full_dir, f"power{i}_input")
                            v_file = os.path.join(full_dir, f"in{i}_input")
                            if os.path.exists(p_file) and os.path.exists(v_file):
                                paths[f"rail_{i}_power"] = p_file
                                paths[f"rail_{i}_voltage"] = v_file
                        if paths:
                            return paths
        except Exception:
            pass
        return None

    def query_hardware_telemetry(self):
        """
        Reads power and voltage from hardware. Routes based on detected platform.
        INA3221 path returns (measured_power_w, measured_voltage_v).
        pynvml path returns (measured_power_w, estimated_voltage_v) — NVML does
        not expose core voltage on most GPUs; voltage is a parasitic-model estimate.
        Raises RuntimeError if no hardware source is reachable.
        """
        if self.jetson_paths:
            try:
                with open(self.jetson_paths["rail_1_power"], "r") as f:
                    power_mw = float(f.read().strip())
                with open(self.jetson_paths["rail_1_voltage"], "r") as f:
                    voltage_mv = float(f.read().strip())
                return power_mw / 1000.0, voltage_mv / 1000.0
            except Exception as e:
                raise RuntimeError(f"INA3221 sysfs read failed: {e}")

        if self._has_pynvml and self._nvml_handle is not None:
            try:
                import pynvml
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                power_w = power_mw / 1000.0
                voltage_est = 1.0 - (power_w / 1.0) * 0.005
                return power_w, voltage_est
            except Exception as e:
                raise RuntimeError(f"pynvml power query failed: {e}")

        raise RuntimeError(
            "No hardware telemetry source available. "
            "Requires Jetson INA3221 sysfs nodes or pynvml."
        )

    def apply_kalman_filter(self, measured_power):
        p_pred = self.p_est + self.Q
        k_gain = p_pred / (p_pred + self.R_cov)
        self.x_est = self.x_est + k_gain * (measured_power - self.x_est)
        self.p_est = (1.0 - k_gain) * p_pred
        return self.x_est

    def update_buffers(self, t_curr, power, voltage):
        self.time_buffer.append(t_curr)
        self.power_buffer.append(power)
        self.voltage_buffer.append(voltage)

        if len(self.power_buffer) > self.buffer_len:
            self.time_buffer.pop(0)
            self.power_buffer.pop(0)
            self.voltage_buffer.pop(0)

    def analyze_spectrum(self, threshold_db=-25.0):
        """
        Detects anomalous power spectral energy in the sub-Nyquist band.
        At the telemetry sampling rate (~20 Hz), this identifies power envelope
        modulation (elevated mean, increased variance, step transitions between
        phases), not the kHz-scale attack modulation frequency itself.
        Returns (is_anomalous, peak_psd_db, nyquist_hz).
        """
        nyquist_hz = self.sampling_rate_hz / 2.0

        if len(self.power_buffer) < 64:
            return False, 0.0, nyquist_hz

        power_array = np.array(self.power_buffer) - np.mean(self.power_buffer)
        freqs, psd = signal.welch(power_array, fs=self.sampling_rate_hz, nperseg=min(len(power_array), 128))

        target_band = (freqs >= 2.0)
        if not np.any(target_band):
            return False, 0.0, nyquist_hz

        max_psd_w = np.max(psd[target_band])
        psd_db = 10 * np.log10(max_psd_w + 1e-10)

        is_anomalous = psd_db > threshold_db
        return is_anomalous, psd_db, nyquist_hz

    def run_realtime_monitor(self, duration_sec=30.0):
        """
        Launches a live command-line monitor reading real hardware telemetry.
        Measures and reports the actual achieved sampling rate at exit.
        """
        print("=====================================================================")
        print("         Project GAIA: Hardware Telemetry Monitor (T2)               ")
        print("=====================================================================")
        print(f"[*] Detected Host System: {self.platform_name}")
        print(f"[*] Target sampling rate: {self.sampling_rate_hz} Hz")
        print(f"[*] Nyquist limit: {self.sampling_rate_hz / 2.0:.1f} Hz (envelope detection only)\n")

        start_time = time.time()
        sample_count = 0
        print(f"{'Elapsed (s)':<12}{'Power (W)':<12}{'Est. Voltage (V)':<16}{'Kalman Pow (W)':<16}{'Anomaly State':<15}")
        print("-" * 75)

        try:
            while time.time() - start_time < duration_sec:
                t_elapsed = time.time() - start_time
                raw_p, raw_v = self.query_hardware_telemetry()
                sample_count += 1

                smoothed_p = self.apply_kalman_filter(raw_p)
                self.update_buffers(t_elapsed, raw_p, raw_v)

                is_attack, peak_db, nyquist_hz = self.analyze_spectrum()
                state_str = "SAFE"
                if is_attack:
                    state_str = "!!! HAZARD !!!" if smoothed_p > 80.0 else "ANOMALY"

                print(f"{t_elapsed:<12.2f}{raw_p:<12.2f}{raw_v:<16.4f}{smoothed_p:<16.2f}{state_str:<15}")
                time.sleep(1.0 / self.sampling_rate_hz)

        except KeyboardInterrupt:
            print("\n[-] Monitoring stopped by user.")

        elapsed = time.time() - start_time
        actual_rate = sample_count / elapsed if elapsed > 0 else 0.0
        print(f"\n[+] Monitor completed: {sample_count} samples in {elapsed:.1f}s")
        print(f"[+] Achieved sampling rate: {actual_rate:.2f} Hz (Nyquist: {actual_rate / 2.0:.2f} Hz)")

    def process_logged_file(self, csv_filepath, output_png=None):
        """
        Batch Mode: Ingests a T1 experimental CSV and runs the complete
        analytical pipeline (Kalman, Spectrogram, PCA, and plotting).

        Expects T1 CSV columns: timestamp, power_w, voltage_v, phase
        """
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Target CSV file not found: {csv_filepath}")

        if output_png is None:
            output_png = os.path.join(SCRIPT_DIR, "gpu_defense_diagnostics_v2.png")

        print(f"[+] Loading dataset: {csv_filepath}")
        df = pd.read_csv(csv_filepath)

        required_cols = ["timestamp", "power_w", "voltage_v", "phase"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Expected T1 output with columns: {required_cols}"
            )

        df.rename(columns={
            "timestamp": "Timestamp",
            "power_w": "Power_W",
            "voltage_v": "Voltage_V",
            "phase": "Label"
        }, inplace=True)

        df["Label_Binary"] = df["Label"].apply(
            lambda x: 1 if "stress" in str(x).lower() or "oversubscribe" in str(x).lower() else 0
        )

        print("[+] Column provenance:")
        print("    - power_w   : MEASURED (hardware sensor)")
        if "data_source" in df.columns:
            sources = df["data_source"].unique()
            for src in sources:
                if "INA3221" in str(src):
                    print("    - voltage_v : MEASURED (Jetson INA3221 board rail)")
                elif "NVML" in str(src):
                    print("    - voltage_v : DERIVED (parasitic model from NVML power, NOT on-die measurement)")
        else:
            print("    - voltage_v : provenance unknown (no data_source column)")
        print("    - phase     : MEASURED (experiment condition label)")

        print("[+] Running Kalman Telemetry Smoothing...")
        filtered = []
        for val in df['Power_W'].values:
            filtered.append(self.apply_kalman_filter(val))
        df['Kalman_Power_W'] = filtered

        print("[+] Executing FFT Spectrogram Generation...")
        dt = df['Timestamp'].diff().mean()
        fs_est = 1.0 / dt if dt > 0 else self.sampling_rate_hz
        print(f"    Estimated Dataset Sampling Rate: {fs_est:.2f} Hz")

        freqs, times, Sxx = signal.spectrogram(
            df['Power_W'].values,
            fs=fs_est,
            nperseg=min(len(df), 128),
            noverlap=min(len(df)//2, 64)
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        print("[+] Executing PCA State-Separability Cluster Fitting...")
        features = []
        labels = []
        window_size = min(len(df) // 10, 50)
        if window_size < 5:
            window_size = 5

        for i in range(0, len(df) - window_size, window_size):
            sub_df = df.iloc[i:i+window_size]
            mean_p = sub_df['Power_W'].mean()
            std_p = sub_df['Power_W'].std()
            min_v = sub_df['Voltage_V'].min()
            p2p_v = sub_df['Voltage_V'].max() - sub_df['Voltage_V'].min()

            features.append([mean_p, std_p, min_v, p2p_v])
            labels.append(1 if sub_df['Label_Binary'].mean() > 0.5 else 0)

        features = np.array(features)
        labels = np.array(labels)

        if len(features) > 2:
            scaler = StandardScaler()
            scaled_feat = scaler.fit_transform(features)
            pca = PCA(n_components=2)
            projected = pca.fit_transform(scaled_feat)
            var_ratio = pca.explained_variance_ratio_
            print(f"    PCA Fit Successful. PC1 Variance: {var_ratio[0]*100:.1f}%, PC2: {var_ratio[1]*100:.1f}%")
        else:
            raise ValueError(
                f"Insufficient data for PCA: only {len(features)} sliding window segments "
                f"(need > 2). Increase experiment duration or reduce window_size."
            )

        self._generate_diagnostic_plots(df, freqs, times, Sxx_db, projected, labels, output_png, fs_est)
        print("[+] Batch processing finished.")

    def _generate_diagnostic_plots(self, df, freqs, times, Sxx_db, projected, pca_labels, output_png, fs_est=20.0):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
        plt.subplots_adjust(hspace=0.4)

        axs[0].plot(df['Timestamp'], df['Power_W'], color='#e74c3c', alpha=0.3, label='Raw Power Draw')
        axs[0].plot(df['Timestamp'], df['Kalman_Power_W'], color='#2c3e50', linewidth=1.5, label='Kalman Filtered')
        axs[0].set_title("GPU Temporal Power Telemetry & Kalman Filtering (Thrust T2)", fontsize=11, fontweight='bold')
        axs[0].set_xlabel("Time (Seconds)", fontsize=9)
        axs[0].set_ylabel("Power (Watts)", fontsize=9)
        axs[0].legend(loc='upper right', fontsize=8)
        axs[0].grid(True, linestyle=':', alpha=0.6)

        pcm = axs[1].pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='viridis')
        axs[1].set_title("Power Spectral Density Spectrogram (Power Envelope Spectral Analysis)", fontsize=11, fontweight='bold')
        axs[1].set_xlabel("Time (Seconds)", fontsize=9)
        axs[1].set_ylabel("Frequency (Hz)", fontsize=9)
        fig.colorbar(pcm, ax=axs[1], label='Power (dB)')
        nyquist_hz = fs_est / 2.0
        axs[1].text(0.01, 0.97, f"Fs={fs_est:.1f} Hz | Nyquist={nyquist_hz:.1f} Hz",
                    transform=axs[1].transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        if len(projected) > 0:
            pca_df = pd.DataFrame({'PC1': projected[:, 0], 'PC2': projected[:, 1], 'State': pca_labels})
            benign = pca_df[pca_df['State'] == 0]
            attack = pca_df[pca_df['State'] == 1]
            axs[2].scatter(benign['PC1'], benign['PC2'], c='#2ecc71', label='Benign Workload', alpha=0.7, edgecolors='none')
            axs[2].scatter(attack['PC1'], attack['PC2'], c='#9b59b6', label='Active Power Attack', alpha=0.7, edgecolors='none')
        axs[2].set_title("PCA Feature Space Projection (Workload State-Separability)", fontsize=11, fontweight='bold')
        axs[2].set_xlabel("Principal Component 1 (PC1)", fontsize=9)
        axs[2].set_ylabel("Principal Component 2 (PC2)", fontsize=9)
        axs[2].legend(fontsize=8)
        axs[2].grid(True, linestyle=':', alpha=0.6)

        plt.savefig(output_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Diagnostic figure saved to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project GAIA: T2 Defense Filters & Anomaly Detection")
    parser.add_argument("--mode", type=str, choices=["monitor", "batch"], default="monitor",
                        help="Operating mode: 'monitor' (live stream) or 'batch' (process CSV)")
    parser.add_argument("--file", type=str, default="", help="Path to T1 experimental CSV (required for batch mode)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path (batch mode)")
    parser.add_argument("--duration", type=float, default=20.0, help="Monitoring duration in seconds (monitor mode)")
    parser.add_argument("--rate", type=float, default=20.0, help="Target telemetry sampling rate in Hz")
    args = parser.parse_args()

    detector = PlatformAgnosticAnomalyDetector(sampling_rate_hz=args.rate)

    if args.mode == "monitor":
        detector.run_realtime_monitor(duration_sec=args.duration)
    elif args.mode == "batch":
        if not args.file:
            raise ValueError("Batch mode requires --file <path_to_t1_csv>")
        detector.process_logged_file(args.file, output_png=args.output)
