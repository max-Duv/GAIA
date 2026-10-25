"""
Project GAIA: Prometheus-Compatible GPU Telemetry Exporter
==========================================================
Lightweight HTTP server that exposes GPU power, clock, temperature,
and utilization metrics in Prometheus exposition format.

Designed as the industry-standard monitoring bridge for Thrust 3:
  - Scrape target for Prometheus (default port 9400)
  - Compatible with Grafana dashboards

Usage:
    python prometheus_exporter.py [--port 9400] [--interval 1.0]

Metrics exported:
    gaia_gpu_power_watts         - Module rail power (INA3221 / NVML)
    gaia_gpu_voltage_volts       - Module rail voltage
    gaia_gpu_clock_mhz           - GPU SM clock frequency
    gaia_gpu_temperature_celsius - GPU die temperature
    gaia_gpu_utilization_percent - GPU compute utilization
    gaia_gpu_memory_utilization_percent - GPU memory utilization
    gaia_anomaly_detected        - Power-level anomaly flag (0/1)
    gaia_mean_power_watts        - Detector sliding-window mean power
"""
import os
import sys
import time
import json
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PARENT_DIR, "T1"))
sys.path.insert(0, os.path.join(PARENT_DIR, "T2"))

from gpu_fault_assessor_v2 import GPUFaultAssessorV2
from gpu_defense_filters_v2 import PlatformAgnosticAnomalyDetector


class TelemetryCollector:
    def __init__(self, detection_threshold_w=60.0):
        self.assessor = GPUFaultAssessorV2()
        self.detector = PlatformAgnosticAnomalyDetector(
            sampling_rate_hz=4.0,
            power_anomaly_threshold_w=detection_threshold_w
        )
        self.lock = threading.Lock()
        self.latest = {}
        self._start_time = time.time()

    def poll(self):
        hw = self.assessor.get_hardware_telemetry()
        t_elapsed = time.time() - self._start_time
        p_w = hw.get("power_w", 0.0)
        v_v = hw.get("voltage_v", 0.0)

        self.detector.update_buffers(t_elapsed, p_w, v_v)
        is_anomalous, mean_power, _ = self.detector.analyze_power_level()

        with self.lock:
            self.latest = {
                "power_w": p_w,
                "voltage_v": v_v,
                "sm_clock_mhz": hw.get("sm_clock_mhz", 0),
                "gpu_temp_c": hw.get("gpu_temp_c", 0),
                "gpu_util_pct": hw.get("gpu_util_pct", 0),
                "mem_util_pct": hw.get("mem_util_pct", 0),
                "anomaly_detected": int(is_anomalous),
                "mean_power_w": mean_power,
            }

    def get_prometheus_metrics(self):
        with self.lock:
            d = dict(self.latest)
        if not d:
            return ""

        lines = [
            "# HELP gaia_gpu_power_watts GPU module rail power draw in watts",
            "# TYPE gaia_gpu_power_watts gauge",
            f"gaia_gpu_power_watts {d.get('power_w', 0):.3f}",
            "# HELP gaia_gpu_voltage_volts GPU module rail voltage",
            "# TYPE gaia_gpu_voltage_volts gauge",
            f"gaia_gpu_voltage_volts {d.get('voltage_v', 0):.5f}",
            "# HELP gaia_gpu_clock_mhz GPU SM clock frequency in MHz",
            "# TYPE gaia_gpu_clock_mhz gauge",
            f"gaia_gpu_clock_mhz {d.get('sm_clock_mhz', 0)}",
            "# HELP gaia_gpu_temperature_celsius GPU die temperature",
            "# TYPE gaia_gpu_temperature_celsius gauge",
            f"gaia_gpu_temperature_celsius {d.get('gpu_temp_c', 0)}",
            "# HELP gaia_gpu_utilization_percent GPU compute utilization",
            "# TYPE gaia_gpu_utilization_percent gauge",
            f"gaia_gpu_utilization_percent {d.get('gpu_util_pct', 0)}",
            "# HELP gaia_gpu_memory_utilization_percent GPU memory utilization",
            "# TYPE gaia_gpu_memory_utilization_percent gauge",
            f"gaia_gpu_memory_utilization_percent {d.get('mem_util_pct', 0)}",
            "# HELP gaia_anomaly_detected Power-level anomaly detector flag",
            "# TYPE gaia_anomaly_detected gauge",
            f"gaia_anomaly_detected {d.get('anomaly_detected', 0)}",
            "# HELP gaia_mean_power_watts Detector sliding-window mean power",
            "# TYPE gaia_mean_power_watts gauge",
            f"gaia_mean_power_watts {d.get('mean_power_w', 0):.3f}",
        ]
        return "\n".join(lines) + "\n"

    def get_json_metrics(self):
        with self.lock:
            return json.dumps(self.latest, indent=2)


collector = None


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = collector.get_prometheus_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        elif self.path == "/json":
            body = collector.get_json_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>"
                             b"<h1>Project GAIA: GPU Telemetry Exporter</h1>"
                             b"<p><a href='/metrics'>Prometheus Metrics</a></p>"
                             b"<p><a href='/json'>JSON Metrics</a></p>"
                             b"</body></html>")

    def log_message(self, format, *args):
        pass


def polling_loop(interval):
    while True:
        try:
            collector.poll()
        except Exception as e:
            print(f"[exporter] Poll error: {e}")
        time.sleep(interval)


def main():
    global collector

    parser = argparse.ArgumentParser(description="GAIA Prometheus GPU Telemetry Exporter")
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--det-threshold", type=float, default=60.0,
                        help="Anomaly detection threshold in watts")
    args = parser.parse_args()

    print("=====================================================================")
    print("  Project GAIA: Prometheus-Compatible GPU Telemetry Exporter")
    print(f"  Port: {args.port} | Poll interval: {args.interval}s")
    print(f"  Detection threshold: {args.det_threshold}W")
    print("=====================================================================")

    collector = TelemetryCollector(detection_threshold_w=args.det_threshold)

    poller = threading.Thread(target=polling_loop, args=(args.interval,), daemon=True)
    poller.start()

    print(f"[exporter] Polling started ({args.interval}s interval)")
    print(f"[exporter] Prometheus endpoint: http://0.0.0.0:{args.port}/metrics")
    print(f"[exporter] JSON endpoint:       http://0.0.0.0:{args.port}/json")

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[exporter] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
