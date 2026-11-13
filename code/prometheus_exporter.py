#!/usr/bin/env python3
"""GAIA Prometheus Exporter — exposes GPU power-domain telemetry as
Prometheus metrics from Jetson AGX Thor INA3221 sensors, devfreq,
and thermal sysfs nodes.

Usage:
    pip install prometheus_client
    python prometheus_exporter.py [--port 9101] [--interval 0.03125] [--threshold 60]

On non-Jetson platforms the exporter serves simulated data with a
warning logged at startup.
"""

import argparse
import glob
import logging
import math
import os
import random
import sys
import threading
import time

from prometheus_client import (
    Counter,
    Gauge,
    Info,
    start_http_server,
)

LOG = logging.getLogger("gaia_exporter")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

POWER = Gauge(
    "gaia_gpu_power_watts",
    "Total module-rail power (sum of all INA3221 channels)",
)
VOLTAGE = Gauge(
    "gaia_gpu_voltage_volts",
    "Module-rail bus voltage from INA3221",
    ["channel"],
)
CURRENT = Gauge(
    "gaia_gpu_current_amps",
    "Per-channel current from INA3221",
    ["channel"],
)
CLOCK = Gauge(
    "gaia_gpu_clock_hz",
    "GPU core clock frequency from devfreq",
)
TEMP = Gauge(
    "gaia_gpu_temperature_celsius",
    "GPU thermal zone temperature",
)
UTIL = Gauge(
    "gaia_gpu_utilization_ratio",
    "GPU utilization (0.0–1.0) from /sys/devices/gpu.0/load",
)
SLEW = Gauge(
    "gaia_gpu_current_slew_rate",
    "Absolute current slew rate |dI/dt| in A/s",
)
ANOMALY = Gauge(
    "gaia_anomaly_detected",
    "1 when mean power exceeds the configured threshold, else 0",
)
SAMPLES = Counter(
    "gaia_telemetry_samples_total",
    "Total telemetry samples collected since exporter start",
)
BUILD = Info(
    "gaia_exporter",
    "GAIA exporter build metadata",
)
BUILD.info({
    "version": "1.0.0",
    "platform": "jetson_agx_thor",
    "sensor": "ina3221",
})

# ---------------------------------------------------------------------------
# Sysfs helpers (Jetson AGX Thor)
# ---------------------------------------------------------------------------

def _discover_ina3221_channels():
    """Return list of dicts with keys: label, voltage_path, current_path,
    power_path for each INA3221 channel found via hwmon."""
    channels = []
    for h_path in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name_file = os.path.join(h_path, "name")
        drv_name = ""
        if os.path.exists(name_file):
            try:
                with open(name_file) as f:
                    drv_name = f.read().strip()
            except OSError:
                pass
        if "ina3221" not in drv_name.lower():
            continue

        for i in range(1, 4):
            label = f"rail_{i}"
            label_file = os.path.join(h_path, f"in{i}_label")
            if os.path.exists(label_file):
                try:
                    with open(label_file) as f:
                        label = f.read().strip()
                except OSError:
                    pass

            ch = {"label": label}
            for kind, prefix in [("voltage", "in"), ("current", "curr"), ("power", "power")]:
                p = os.path.join(h_path, f"{prefix}{i}_input")
                if os.path.exists(p):
                    ch[f"{kind}_path"] = p
            if any(k.endswith("_path") for k in ch):
                channels.append(ch)
    return channels


def _find_devfreq():
    path = "/sys/class/devfreq/gpu-gpc-0"
    return path if os.path.isdir(path) else None


def _find_thermal():
    for tz in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(os.path.join(tz, "type")) as f:
                if "gpu" in f.read().strip().lower():
                    return tz
        except OSError:
            continue
    return None


GPU_LOAD_PATH = "/sys/devices/gpu.0/load"


def _read_sysfs_float(path):
    try:
        with open(path) as f:
            raw = f.read()
        if raw:
            return float(raw.strip())
    except (OSError, ValueError, TypeError):
        pass
    return None

# ---------------------------------------------------------------------------
# Collector: real Jetson
# ---------------------------------------------------------------------------

class JetsonCollector:
    """Reads INA3221, devfreq, thermal, and GPU load from sysfs."""

    def __init__(self, threshold_w):
        self.channels = _discover_ina3221_channels()
        self.devfreq = _find_devfreq()
        self.thermal = _find_thermal()
        self.threshold_w = threshold_w
        self._prev_current_a = None
        self._prev_time = None
        LOG.info(
            "JetsonCollector: %d INA3221 channels, devfreq=%s, thermal=%s",
            len(self.channels), self.devfreq, self.thermal,
        )

    def collect(self):
        now = time.monotonic()
        total_power_w = 0.0
        total_current_a = 0.0

        for ch in self.channels:
            label = ch["label"]
            v_path = ch.get("voltage_path")
            c_path = ch.get("current_path")
            p_path = ch.get("power_path")

            if v_path:
                val = _read_sysfs_float(v_path)
                if val is not None:
                    VOLTAGE.labels(channel=label).set(val / 1000.0)

            if c_path:
                val = _read_sysfs_float(c_path)
                if val is not None:
                    current_a = val / 1000.0
                    CURRENT.labels(channel=label).set(current_a)
                    total_current_a += current_a

            if p_path:
                val = _read_sysfs_float(p_path)
                if val is not None:
                    total_power_w += val / 1e6

        POWER.set(total_power_w)

        if self.devfreq:
            val = _read_sysfs_float(os.path.join(self.devfreq, "cur_freq"))
            if val is not None:
                CLOCK.set(val * 1000.0)

        if self.thermal:
            val = _read_sysfs_float(os.path.join(self.thermal, "temp"))
            if val is not None:
                TEMP.set(val / 1000.0)

        if os.path.exists(GPU_LOAD_PATH):
            val = _read_sysfs_float(GPU_LOAD_PATH)
            if val is not None:
                UTIL.set(val / 1000.0)

        if self._prev_current_a is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0:
                SLEW.set(abs(total_current_a - self._prev_current_a) / dt)
        self._prev_current_a = total_current_a
        self._prev_time = now

        ANOMALY.set(1.0 if total_power_w > self.threshold_w else 0.0)
        SAMPLES.inc()

# ---------------------------------------------------------------------------
# Collector: simulated (non-Jetson platforms)
# ---------------------------------------------------------------------------

class SimulatedCollector:
    """Generates plausible synthetic telemetry for development/testing."""

    def __init__(self, threshold_w):
        self.threshold_w = threshold_w
        self._prev_current = 2.3
        self._t0 = time.monotonic()
        self._prev_time = self._t0
        LOG.warning(
            "No INA3221 sensors found — running with SIMULATED data. "
            "Metrics are synthetic and not from real hardware."
        )

    def collect(self):
        now = time.monotonic()
        elapsed = now - self._t0

        base_power = 45.0 + 3.0 * math.sin(elapsed * 0.1)
        attack_burst = 40.0 if (int(elapsed) % 60) > 30 else 0.0
        power = base_power + attack_burst + random.gauss(0, 0.5)

        voltage = 19.75 + random.gauss(0, 0.005)
        current = power / voltage

        POWER.set(power)
        VOLTAGE.labels(channel="simulated_rail_1").set(voltage)
        CURRENT.labels(channel="simulated_rail_1").set(current)
        CLOCK.set(1.575e9 if attack_burst > 0 else 1.05e9)
        TEMP.set(42.0 + power * 0.1 + random.gauss(0, 0.2))
        UTIL.set(min(1.0, 0.25 + attack_burst / 80.0 + random.gauss(0, 0.02)))

        dt = now - self._prev_time
        if dt > 0:
            SLEW.set(abs(current - self._prev_current) / dt)
        self._prev_current = current
        self._prev_time = now

        ANOMALY.set(1.0 if power > self.threshold_w else 0.0)
        SAMPLES.inc()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _collection_loop(collector, interval):
    while True:
        try:
            collector.collect()
        except Exception:
            LOG.exception("Collection error")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="GAIA Prometheus Exporter")
    parser.add_argument("--port", type=int, default=9101,
                        help="HTTP port for /metrics (default: 9101)")
    parser.add_argument("--interval", type=float, default=1.0 / 32.0,
                        help="Seconds between samples (default: 0.03125 = 32 Hz)")
    parser.add_argument("--threshold", type=float, default=60.0,
                        help="Anomaly power threshold in watts (default: 60)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    channels = _discover_ina3221_channels()
    if channels:
        collector = JetsonCollector(args.threshold)
    else:
        collector = SimulatedCollector(args.threshold)

    start_http_server(args.port)
    LOG.info("Serving metrics on http://0.0.0.0:%d/metrics", args.port)
    LOG.info("Sampling at %.1f Hz, anomaly threshold = %.1f W",
             1.0 / args.interval, args.threshold)

    _collection_loop(collector, args.interval)


if __name__ == "__main__":
    main()
