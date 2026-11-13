# GAIA: Performance Counter Collection on Jetson AGX Thor

**Purpose:** Close the last proposal gap — T1/T2 "performance counters" and "low-level profiler."
**Proposal Figure 1, item ⑥:** "hardware performance counters"

## Step 1: Probe available tools

SSH into the Jetson and run:

```bash
cd ~/GAIA-USENIX
git pull
sudo env "PATH=$PATH" python3 code/T2/cupti_perf_counters.py --probe --out results/cupti_probe
```

This will report which profiling tools are available:
- **ncu** (Nsight Compute) — BEST option, per-kernel L2 miss, SM occupancy, DRAM BW
- **nsys** (Nsight Systems) — GOOD option, GPU-wide utilization at 100Hz+
- **tegrastats** — FALLBACK, GR3D_FREQ%, EMC%, GPU power (Jetson-specific)

## Step 2a: If ncu is available (BEST — per-kernel counters)

Run a short attack experiment under ncu profiling:

```bash
# Profile just the attacker + victim (sustained stress, 1 trial, 15s)
sudo env "PATH=$PATH" python3 code/T2/cupti_perf_counters.py \
    --ncu-cmd "python3 code/T1/controlled_3phase_runner_v2.py --trials 1 --duration 15 --conditions victim_alone victim_sustained_stress --out results/cupti_ncu" \
    --out results/cupti_ncu
```

This collects per-kernel: L2 cache read/write sectors, SM warp occupancy, DRAM throughput.

**Note:** ncu significantly slows kernel execution (10-100x). Run a SHORT experiment (1 trial, 15s phases).

## Step 2b: If nsys is available (GOOD — GPU-wide metrics)

```bash
sudo env "PATH=$PATH" python3 code/T2/cupti_perf_counters.py \
    --profile-cmd "python3 code/T1/controlled_3phase_runner_v2.py --trials 1 --duration 15 --conditions victim_alone victim_sustained_stress --out results/cupti_nsys" \
    --out results/cupti_nsys
```

This captures GPU utilization metrics at 100Hz via nsys's built-in GPU metrics.

## Step 2c: If only tegrastats (FALLBACK)

Run tegrastats alongside a normal experiment:

**Terminal 1 (experiment):**
```bash
sudo env "PATH=$PATH" python3 code/T1/controlled_3phase_runner_v2.py \
    --trials 3 --duration 15 \
    --conditions victim_alone victim_sustained_stress victim_cache_only victim_compute_only \
    --out results/cupti_tegra
```

**Terminal 2 (tegrastats collection, start BEFORE Terminal 1):**
```bash
sudo python3 code/T2/cupti_perf_counters.py \
    --tegrastats --duration 300 --interval 100 --out results/cupti_tegra
```

tegrastats provides:
- `GR3D_FREQ%` — GPU utilization (comparable to gpu_util_pct from sysfs)
- `EMC_FREQ%` — memory controller utilization (**NEW — not in current features**)
- GPU power breakdown by rail
- CPU/GPU temperature

## Step 3: Bring results back

Copy results to the Windows machine:

```bash
# From MobaXterm or local machine:
scp -r jetson:~/GAIA-USENIX/results/cupti_* results/
```

## What the data adds to the paper

| Metric | Source | Proposal commitment | Paper status |
|---|---|---|---|
| L2 cache miss rate | ncu | "cache lines" (T2) | MISSING |
| SM warp occupancy | ncu/nsys | "performance counters" (T1/T2) | MISSING |
| DRAM bandwidth | ncu | "low-level profiler" (T2) | MISSING |
| Memory controller util | tegrastats EMC_FREQ% | implicit in profiler | MISSING |
| GPU compute util | tegrastats/nsys | Already have via sysfs | ✅ DONE |

Even tegrastats (the fallback) adds EMC_FREQ% — a memory-controller utilization metric not in the current 8 features. This strengthens the "low-level profiler" claim.

## What happens next

Once the data is back, we'll:
1. Add the new metrics as features to the detector (9-12 features)
2. Re-evaluate classifier with expanded feature set
3. Write a short paragraph in Section 6.2 reporting the improvement
4. Update the proposal alignment to mark T1/T2 perf counter gap as CLOSED
