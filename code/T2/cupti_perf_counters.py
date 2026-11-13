"""CUPTI-based performance counter collection for GAIA detector enrichment.

Closes the proposal T1/T2 gap: "performance counters" and "low-level profiler
that leverages administrative privileges to monitor cache lines, performance
counters, and kernel power."

Approach: Use NVIDIA Nsight Systems (nsys) CLI profiling to collect GPU
performance metrics during attack experiments. This avoids direct CUPTI API
integration (which requires C-level hooks) and instead uses nsys's built-in
profiling with sqlite output for post-hoc analysis.

If nsys is unavailable, falls back to tegrastats parsing on Jetson.

Usage:
    # Profile an existing experiment run:
    python cupti_perf_counters.py --profile-cmd "python controlled_3phase_runner_v2.py ..." --out results_cupti/

    # Parse an existing nsys report:
    python cupti_perf_counters.py --parse results_cupti/report.sqlite --out results_cupti/

    # Collect tegrastats fallback:
    python cupti_perf_counters.py --tegrastats --duration 60 --out results_cupti/
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def check_nsys():
    """Check if nsys (Nsight Systems CLI) is available."""
    nsys = shutil.which("nsys")
    if nsys:
        result = subprocess.run([nsys, "--version"], capture_output=True, text=True)
        version = result.stdout.strip() or result.stderr.strip()
        return nsys, version
    return None, None


def check_ncu():
    """Check if ncu (Nsight Compute CLI) is available for per-kernel counters."""
    ncu = shutil.which("ncu")
    if ncu:
        result = subprocess.run([ncu, "--version"], capture_output=True, text=True)
        version = result.stdout.strip() or result.stderr.strip()
        return ncu, version
    return None, None


def check_tegrastats():
    """Check if tegrastats is available (Jetson-specific)."""
    ts = shutil.which("tegrastats")
    return ts is not None


def run_nsys_profile(cmd, out_dir, duration=None):
    """Run a command under nsys profiling, collecting GPU metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_base = out_dir / "nsys_report"

    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--gpu-metrics-device=all",
        "--gpu-metrics-frequency=100",
        f"--output={report_base}",
        "--export=sqlite",
        "--force-overwrite=true",
    ]
    if duration:
        nsys_cmd.extend(["--duration", str(duration)])

    nsys_cmd.extend(cmd if isinstance(cmd, list) else cmd.split())

    print(f"Running nsys profile: {' '.join(nsys_cmd)}")
    result = subprocess.run(nsys_cmd, capture_output=False, text=True)

    sqlite_path = report_base.with_suffix(".sqlite")
    if sqlite_path.exists():
        print(f"Report saved: {sqlite_path}")
        return str(sqlite_path)
    else:
        qdrep = report_base.with_suffix(".nsys-rep")
        if qdrep.exists():
            subprocess.run(["nsys", "export", "--type=sqlite", str(qdrep)])
            if sqlite_path.exists():
                return str(sqlite_path)
    print("WARNING: No sqlite report generated")
    return None


def parse_nsys_sqlite(sqlite_path, out_dir):
    """Extract GPU metrics from nsys sqlite report into CSV."""
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU kernel execution summary
    try:
        kernels = conn.execute("""
            SELECT
                k.start AS start_ns,
                k.end AS end_ns,
                (k.end - k.start) AS duration_ns,
                s.value AS kernel_name,
                k.gridX, k.gridY, k.gridZ,
                k.blockX, k.blockY, k.blockZ,
                k.registersPerThread,
                k.sharedMemory,
                k.deviceId
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON k.demangledName = s.id
            ORDER BY k.start
        """).fetchall()

        cols = ["start_ns", "end_ns", "duration_ns", "kernel_name",
                "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ",
                "registers_per_thread", "shared_memory_bytes", "device_id"]
        csv_path = out_dir / "cupti_kernels.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(kernels)
        print(f"Kernel trace: {len(kernels)} rows -> {csv_path}")
    except Exception as e:
        print(f"Kernel trace extraction failed: {e}")

    # GPU metrics (sm utilization, memory throughput, etc.)
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        gpu_metric_tables = [t for t in tables if "GPU_METRICS" in t.upper()
                            or "GENERIC_EVENT" in t.upper()]
        for tbl in gpu_metric_tables:
            rows = conn.execute(f"SELECT * FROM {tbl} LIMIT 5").fetchall()
            desc = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            col_names = [d[1] for d in desc]
            print(f"  GPU metrics table '{tbl}': {len(col_names)} cols, sample: {col_names[:8]}")

            all_rows = conn.execute(f"SELECT * FROM {tbl}").fetchall()
            csv_path = out_dir / f"{tbl.lower()}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(col_names)
                w.writerows(all_rows)
            print(f"  -> {len(all_rows)} rows -> {csv_path}")
    except Exception as e:
        print(f"GPU metrics extraction failed: {e}")

    conn.close()


def run_ncu_counters(kernel_name_pattern, out_dir, cmd):
    """Use Nsight Compute to collect per-kernel hardware counters."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ncu_counters.ncu-rep"

    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--metrics",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,"
        "lts__t_sectors_op_read.sum,"
        "lts__t_sectors_op_write.sum,"
        "sm__warps_active.avg.pct_of_peak_sustained_active,"
        "dram__bytes.sum.per_second,"
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "--csv",
        f"--log-file={out_dir / 'ncu_counters.csv'}",
        "-o", str(report_path),
    ]
    if kernel_name_pattern:
        ncu_cmd.extend(["--kernel-name", kernel_name_pattern])

    ncu_cmd.extend(cmd if isinstance(cmd, list) else cmd.split())

    print(f"Running ncu profiling: {' '.join(ncu_cmd[:8])}...")
    result = subprocess.run(ncu_cmd, capture_output=False, text=True)
    csv_path = out_dir / "ncu_counters.csv"
    if csv_path.exists():
        print(f"NCU counters saved: {csv_path}")
    return str(csv_path) if csv_path.exists() else None


def collect_tegrastats(duration, interval_ms, out_dir):
    """Collect tegrastats on Jetson as a lightweight fallback.

    tegrastats provides: GR3D_FREQ (GPU utilization %), GPU power, EMC freq,
    and more. Not as granular as CUPTI but captures utilization metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tegrastats.csv"

    cmd = ["tegrastats", f"--interval", str(interval_ms)]
    print(f"Collecting tegrastats for {duration}s at {interval_ms}ms interval...")

    rows = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.time()
    try:
        while time.time() - start < duration:
            line = proc.stdout.readline().strip()
            if not line:
                continue
            ts = time.time() - start
            parsed = parse_tegrastats_line(line)
            parsed["elapsed_s"] = round(ts, 3)
            rows.append(parsed)
    finally:
        proc.terminate()
        proc.wait()

    if rows:
        keys = rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"tegrastats: {len(rows)} samples -> {csv_path}")
    return str(csv_path)


def parse_tegrastats_line(line):
    """Parse a single tegrastats output line into a dict."""
    result = {"raw": line}

    # GR3D_FREQ XX% — GPU utilization
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        result["gr3d_freq_pct"] = int(m.group(1))

    # EMC_FREQ XX% — memory controller utilization
    m = re.search(r'EMC_FREQ\s+(\d+)%', line)
    if m:
        result["emc_freq_pct"] = int(m.group(1))

    # GPU power: VDD_GPU_SOC XXXX/XXXX or VDD_GPU XXXXmW
    for pattern in [r'VDD_GPU_SOC\s+(\d+)/(\d+)', r'VDD_GPU\s+(\d+)/(\d+)',
                    r'VDD_GPU_SOC\s+(\d+)mW', r'VDD_GPU\s+(\d+)mW']:
        m = re.search(pattern, line)
        if m:
            result["gpu_power_mw"] = int(m.group(1))
            if m.lastindex >= 2:
                result["gpu_power_avg_mw"] = int(m.group(2))
            break

    # Total power: VDD_IN XXXX/XXXX
    for pattern in [r'VDD_IN\s+(\d+)/(\d+)', r'VIN_SYS_5V0\s+(\d+)/(\d+)']:
        m = re.search(pattern, line)
        if m:
            result["total_power_mw"] = int(m.group(1))
            result["total_power_avg_mw"] = int(m.group(2))
            break

    # Temperature: GPU@XX.XC or thermal@XX.XC
    for pattern in [r'GPU@(\d+\.?\d*)C', r'gpu@(\d+\.?\d*)C']:
        m = re.search(pattern, line, re.IGNORECASE)
        if m:
            result["gpu_temp_c"] = float(m.group(1))
            break

    return result


def probe_capabilities():
    """Report what profiling tools are available on this system."""
    print("=" * 60)
    print("GAIA Performance Counter Probe")
    print("=" * 60)

    nsys, nsys_ver = check_nsys()
    ncu, ncu_ver = check_ncu()
    has_tegra = check_tegrastats()

    print(f"\n  nsys (Nsight Systems): {'FOUND ' + nsys_ver if nsys else 'NOT FOUND'}")
    print(f"  ncu  (Nsight Compute): {'FOUND ' + ncu_ver if ncu else 'NOT FOUND'}")
    print(f"  tegrastats (Jetson):   {'FOUND' if has_tegra else 'NOT FOUND'}")

    # Check CUDA version
    nvcc = shutil.which("nvcc")
    if nvcc:
        r = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        m = re.search(r'release (\S+)', r.stdout)
        print(f"  nvcc:                  FOUND ({m.group(1) if m else 'version unknown'})")
    else:
        print("  nvcc:                  NOT FOUND")

    # Check compute capability
    try:
        check_cc = """
import subprocess, re
r = subprocess.run(['python3', '-c',
    'import torch; print(torch.cuda.get_device_capability())'],
    capture_output=True, text=True)
print(r.stdout.strip())
"""
        r = subprocess.run([sys.executable, "-c", check_cc.strip()],
                          capture_output=True, text=True, timeout=10)
        if r.stdout.strip():
            print(f"  Compute capability:    {r.stdout.strip()}")
    except Exception:
        pass

    recs = []
    if ncu:
        recs.append("BEST: Use ncu for per-kernel L2 miss, SM occupancy, DRAM BW")
    if nsys:
        recs.append("GOOD: Use nsys for GPU-wide utilization metrics at 100Hz+")
    if has_tegra:
        recs.append("FALLBACK: Use tegrastats for GR3D_FREQ%, EMC%, GPU power")
    if not recs:
        recs.append("No profiling tools found. Install CUDA toolkit or Nsight tools.")

    print(f"\n  Recommendations:")
    for r in recs:
        print(f"    - {r}")
    print()
    return {"nsys": nsys, "ncu": ncu, "tegrastats": has_tegra}


def main():
    parser = argparse.ArgumentParser(description="GAIA CUPTI performance counter collection")
    parser.add_argument("--probe", action="store_true",
                       help="Probe available profiling tools and exit")
    parser.add_argument("--profile-cmd", type=str,
                       help="Command to profile with nsys")
    parser.add_argument("--ncu-cmd", type=str,
                       help="Command to profile with ncu (per-kernel counters)")
    parser.add_argument("--kernel-name", type=str, default=None,
                       help="Kernel name regex for ncu (default: all)")
    parser.add_argument("--parse", type=str,
                       help="Parse an existing nsys sqlite report")
    parser.add_argument("--tegrastats", action="store_true",
                       help="Collect tegrastats (Jetson fallback)")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration in seconds for tegrastats collection")
    parser.add_argument("--interval", type=int, default=100,
                       help="Interval in ms for tegrastats (default 100)")
    parser.add_argument("--out", type=str, default="results_cupti",
                       help="Output directory")
    args = parser.parse_args()

    if args.probe:
        caps = probe_capabilities()
        json_path = Path(args.out) / "probe_results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(caps, indent=2))
        print(f"Probe results saved: {json_path}")
        return

    if args.profile_cmd:
        sqlite = run_nsys_profile(args.profile_cmd, args.out)
        if sqlite:
            parse_nsys_sqlite(sqlite, args.out)
        return

    if args.ncu_cmd:
        run_ncu_counters(args.kernel_name, args.out, args.ncu_cmd)
        return

    if args.parse:
        parse_nsys_sqlite(args.parse, args.out)
        return

    if args.tegrastats:
        collect_tegrastats(args.duration, args.interval, args.out)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
