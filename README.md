# Project GAIA: Securing AI Data Centers Against Power Attacks

## Revised paper figures

Generate the DAC-oriented figure set while preserving earlier exports:

```bash
python analyze_results.py results_20261024_233655 --figures-only --out-dir results_20261024_233655/figures_dac
```

`publication_figures.py` owns the rendering; the normal analysis command uses it
as well. Replace the results directory for future runs. Outputs include vector
PDFs with embedded TrueType fonts, 300-dpi PNG previews, and caption text files.
Every figure has an exact 7.16-inch canvas width, intended for **two-column**
placement. Use the PDF in the manuscript, not a screenshot of the PNG:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig5_t1_sweep.pdf}
  \caption{Adapt the accompanying caption text to your experiment.}
  \label{fig:parameter-sweep}
\end{figure*}
```

Do not shrink these composite figures to one column: that also shrinks their
9-point labels. Arial is preferred when installed; DejaVu Sans is the portable
fallback. Captions belong in LaTeX, so there are no redundant figure-wide titles.

- Figure 1: power envelopes, using per-trial 0.5-second bin means followed by
  across-trial median and interquartile range. The band is **not** a confidence
  interval. No interpolation bridges missing bins or trial boundaries.
- Figures 2–4 and 6: trial-level comparisons and telemetry/response diagnostics.
  Open circles denote trials; diamonds and error bars summarize trial means
  with 95% Student-t intervals. Feature plots also use distinct marker shapes.
- Figure 5: frequency/window and duty-cycle sweeps. Power and kernel time have
  separate panels and units. Supports both older window sweeps and new sweeps.
- Figure 7: trial mean and minimum rail-voltage summaries, separated from power
  traces to preserve readable labels and expose quantization.

The summary plots retain all trial observations; cosmetic changes do not remove
outliers or change statistical reports. Binned envelopes intentionally suppress
sample-level detail; consult raw telemetry for transient analysis. Inspect final
placement in the compiled manuscript before submission. An optional `--tdp`
adds a user-supplied power reference; no hardware threshold is assumed.

## How This Code Maps to the Research Proposal

The proposal poses the research question: *How can power attacks on GPU hardware in modern AI data centers be designed, evaluated, detected, and mitigated to prevent grid disruptions?*

The project answers this through three tasks (mapped to three code thrusts):

| Proposal Task | Code Thrust | What It Produces |
|---|---|---|
| **Task 1**: Design cache-eviction and occupancy-based GPU microarchitectural attacks | **T1/** | Attack kernels, telemetry traces, victim degradation measurements |
| **Task 2**: Develop defensive monitoring for power attacks | **T2/** | Kalman filter, STFT spectrogram, PCA state separation, anomaly detector |
| **Task 3**: End-to-end monitoring pipeline | **T3/** | Closed-loop mitigation with real-time detection triggering power clamping |

### Proposal Deliverables Produced by This Code

| Deliverable | What to Collect | Source |
|---|---|---|
| **D1**: GPU stressor microbenchmarks, telemetry traces, reproducible evaluation harnesses | `T1/controlled_3phase_experiment_v2.csv`, compiled CUDA binaries, 4-panel IEEE figure | T1 pipeline |
| **D2**: Defensive mechanisms for detecting and countering adversarial GPU kernels | `T2/gpu_defense_diagnostics_v2.png` (Kalman + spectrogram + PCA), anomaly classification stats | T2 pipeline |
| **D3**: End-to-end monitoring prototype to flag anomalous compute patterns | `T3/closed_loop_mitigation_v2.csv`, mitigation effectiveness decomposition | T3 pipeline |
| **D4**: Publication + grant proposal | Aggregate all CSV/PNG outputs for paper figures | All thrusts |

---

## Directory Layout

```
GAIA/
├── T1/                                Thrust 1: Attack & Fault Assessment
│   ├── gpu_power_poc_v3.cu              CUDA stress kernels (PTX cache-eviction + INT32 saturation)
│   ├── gpu_victim_kernel.cu             CUDA victim workload (FMA-bound, CUDA-event timed)
│   ├── gpu_fault_assessor_v2.py         Hardware telemetry (INA3221 + pynvml) and victim dispatch
│   ├── controlled_3phase_runner_v2.py   3-phase experiment harness with repeated trials
│   └── power_analysis_plotter.py        4-panel IEEE publication figure generator
│
├── T2/                                Thrust 2: Signal Processing & Anomaly Detection
│   └── gpu_defense_filters_v2.py        Kalman filter, STFT, Welch PSD, PCA separability
│
├── T3/                                Thrust 3: Closed-Loop Active Mitigation
│   └── closed_loop_mitigator_v2.py      4-phase mitigation experiment with decomposition
│
├── scripts/
│   └── jetson_cleanup.sh               Post-experiment Jetson wipe (repo, caches, history)
│
├── gaia_master_suite_v2.py            Orchestrates T1 → T2 → T3 end-to-end
├── power_analysis_plotter.py          T1 post-hoc plotting (imported by master suite)
└── README.md                          This file
```

---

## Prerequisites

### Hardware

- **NVIDIA Jetson Orin** (primary target) with INA3221 sysfs nodes for board-level voltage/current
- Or any NVIDIA GPU visible to `nvidia-smi` (NVML telemetry only -- no board-rail voltage)

### Software

| Package | Version | Why |
|---|---|---|
| CUDA Toolkit (`nvcc`) | >= 11.0 | Compiles stress and victim kernels |
| `nvidia-smi` | (bundled with driver) | Telemetry, arch detection, power limit control |
| Python | >= 3.8 | Experiment harnesses |
| PyTorch (CUDA) | >= 2.0 | Victim GEMM workload |
| pynvml (`nvidia-ml-py`) | >= 12.0 | Direct NVML telemetry at ~100 Hz |
| NumPy | >= 1.20 | Numerical computation |
| Pandas | >= 1.3 | CSV I/O and data frames |
| SciPy | >= 1.7 | Welch PSD, spectrogram, signal processing |
| scikit-learn | >= 1.0 | PCA and StandardScaler |
| Matplotlib | >= 3.5 | Publication-grade figures |

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib nvidia-ml-py
```

---

## Step-by-Step Experiment Execution

Run all commands from this `GAIA/` directory. The thrusts are sequential -- each produces data consumed by the next.

---

### Step 1: Compile CUDA Backends

**Proposal mapping:** Task 1 -- builds the adversarial kernels (cache-eviction and compute saturation) described in Section 3, Figure 2.

```bash
cd T1
python controlled_3phase_runner_v2.py --duration 1
cd ..
```

This auto-detects the GPU architecture via `nvidia-smi` and compiles `gpu_power_poc_v3.cu` and `gpu_victim_kernel.cu`. The short `--duration 1` confirms compilation without running a full experiment.

**What gets compiled:**
- `gpu_power_poc_v3` -- adversarial stress binary with three modes: `cache` (L2 thrashing via PTX `st.global.wt`/`ld.global.cg`), `compute` (INT32 saturation), `oversubscribe` (both simultaneously across multiple streams)
- `gpu_victim_kernel` -- FMA-bound victim workload timed via CUDA events

---

### Step 2: Verify Hardware Telemetry

**Proposal mapping:** Confirms that the telemetry infrastructure is working before collecting data. The proposal requires real hardware measurements (power variance, voltage sags, throttling events) -- no simulated fallbacks.

```bash
python T1/gpu_fault_assessor_v2.py
```

**Expected output:** A telemetry dict (`power_w`, `voltage_v`, `provenance`, throttle flags) and a victim workload dict (`total_latency_ms`, `gpu_compute_ms`, `checksum`). Both must return real hardware data. If either source is missing, you get a `RuntimeError`.

**Metrics verified here:**
- Power draw in watts (via INA3221 on Jetson or NVML on desktop)
- Voltage (measured on Jetson, parasitic model on desktop)
- Throttle flags (`sw_power_cap`, `hw_slowdown`)
- Victim latency decomposition (total vs GPU compute vs launch delay)

---

### Step 3: Run T1 -- 3-Phase Attack Experiment

**Proposal mapping:** Task 1 core experiment. The three phases directly test the proposal's hypothesis that *modulated pulsing* (not just sustained power draw) is the attack mechanism. This produces the data for the proposal's effectiveness metrics: power variance, peak-induced power, induced throttling, and victim workload slowdown.

```bash
python T1/controlled_3phase_runner_v2.py --duration 15 --trials 10
```

**The three experimental conditions:**

| Phase | What Runs | Proposal Connection |
|---|---|---|
| `victim_alone` | Victim workload, no stress | Clean baseline: uncontested GPU latency and idle power |
| `victim_matched_benign` | Victim + continuous compute stress (same energy, no transients) | Controls for resource contention without transients |
| `victim_pulsed_stress` | Victim + 10 kHz modulated cache + compute (sharp di/dt) | The actual attack: kHz-modulated pulsing per proposal Section 3 |

**Output:** `T1/controlled_3phase_experiment_v2.csv`

**Metrics collected per sample (maps to proposal Section 3, Task 1 metrics):**

| Metric | Column | Proposal Metric |
|---|---|---|
| Power draw | `power_w` | "power variance" |
| Voltage | `voltage_v` | "voltage sags" (PDN droop via V = V_nom - IR - L di/dt) |
| Current rate of change | `di_dt_amps_per_sec` | "transient modulation" (at telemetry rate, not microsecond) |
| Victim total latency | `victim_total_latency_ms` | "workload slowdown" |
| Victim GPU compute time | `victim_gpu_compute_ms` | GPU-only degradation |
| Victim launch delay | `victim_launch_delay_ms` | Scheduler contention |
| Software power cap | `sw_power_cap` | "induced throttling" |
| Hardware slowdown | `hw_slowdown` | "induced throttling" |
| Throttle bitmask | `throttle_bitmask` | Combined throttle state |

**Statistical output (with `--trials N`):**
- Per-phase mean +/- std for latency, power, voltage across trials
- Paired t-test: `victim_pulsed_stress` vs `victim_matched_benign` (p-value + Cohen's d)
- The p < 0.05 threshold for the pulsed-vs-benign contrast is the publishable finding

**Optional mechanism-isolation sweep:**
```bash
python T1/controlled_3phase_runner_v2.py --duration 15 --trials 5 --sweep
```
Adds 7 phases varying pulse window from 1ms to 500ms. Useful for isolating which pulse parameters cause the most degradation.

**Expected runtime:** ~15 min for 10 trials x 3 phases x 15s = 450s of GPU time, plus overhead.

---

### Step 4: Generate T1 Publication Figure (Optional)

**Proposal mapping:** Produces a 4-panel IEEE-format figure for the paper (D4).

```bash
python T1/power_analysis_plotter.py T1/controlled_3phase_experiment_v2.csv 80
```

(Replace `80` with your GPU's TDP in watts, e.g., `30` for Jetson Orin Nano.)

**Output:** A PNG file with four panels:
1. **(a)** Power draw vs 120% oversubscription threshold
2. **(b)** Core voltage sags (V_core = V_nom - IR - L di/dt)
3. **(c)** di/dt switching transients
4. **(d)** Power spectral density (resonant frequency identification)

---

### Step 5: Run T2 -- Batch Signal Processing & Anomaly Detection

**Proposal mapping:** Task 2 -- defensive monitoring. Takes T1's raw telemetry and evaluates whether the attack is detectable from power/voltage features alone. The proposal calls for FFT-based frequency-domain analysis, PCA, and Kalman filters evaluated on precision/recall, false-positive rate, and detection latency.

```bash
python T2/gpu_defense_filters_v2.py --mode batch --file T1/controlled_3phase_experiment_v2.csv
```

**Analytical pipeline (maps to proposal Section 3, Task 2):**

| Stage | Method | Proposal Connection |
|---|---|---|
| 1. Kalman Smoothing | 1D recursive Kalman filter on raw power | "Kalman filters" for noise removal |
| 2. STFT Spectrogram | `scipy.signal.spectrogram` on power signal | "FFT-based frequency-domain analysis" |
| 3. Welch PSD Anomaly | Sliding-window PSD in 2+ Hz band, -25 dB threshold | Resonant attack detection (SAFE / RESONANT ATTACK / HAZARD) |
| 4. PCA State Separation | 4-feature sliding window projected to 2D | "Principal component analysis (PCA)" for state separability |

**PCA features extracted:**
- Mean power (per window)
- Power standard deviation
- Minimum voltage
- Voltage peak-to-peak range

**Output:** `T2/gpu_defense_diagnostics_v2.png` -- 3-panel diagnostic figure:
- (a) Raw vs Kalman-smoothed power trace
- (b) Time-frequency spectrogram
- (c) PCA feature space (green = benign, purple = attack)

**What to look for:** Clean cluster separation in panel (c) means the attack produces a measurably different power/voltage signature -- confirming detectability (D2).

**Important provenance note:** Telemetry via pynvml samples at ~100 Hz (Nyquist = 50 Hz). The attack modulates at 10 kHz. The spectrogram observes the *power envelope* signature, not the kHz modulation directly. Describe this in publications as "power envelope spectral analysis."

---

### Step 6: Run T2 -- Live Monitor (Optional)

**Proposal mapping:** Task 2 real-time detection capability. Demonstrates that the detector works on live telemetry, not just post-hoc CSV analysis.

Terminal 1 -- start the attack:
```bash
T1/gpu_power_poc_v3 --mode oversubscribe --freq 10.0 --duty 0.5 --duration 30
```

Terminal 2 -- start the monitor:
```bash
python T2/gpu_defense_filters_v2.py --mode monitor --duration 30 --rate 20
```

Shows real-time anomaly classification: `SAFE` / `RESONANT ATTACK` / `!!! HAZARD !!!`

---

### Step 7: Run T3 -- Closed-Loop Mitigation

**Proposal mapping:** Task 3 -- end-to-end monitoring pipeline. Combines T1's attack with T2's detector to evaluate whether real-time power clamping can mitigate victim degradation. The 4-phase design answers the key question: *is the latency improvement because the attack is mitigated, or because the GPU is running slower?*

```bash
sudo python T3/closed_loop_mitigator_v2.py --duration 15
```

**`sudo` required** -- `nvidia-smi -pl` needs root to set hardware power limits.

**The four experimental phases:**

| Phase | Stress | Power Limit | What It Measures |
|---|---|---|---|
| `attack_unmitigated` | 10 kHz pulsed | Original | Baseline damage (how bad is the attack?) |
| `attack_mitigated` | 10 kHz pulsed | Clamped on anomaly detection | Mitigation effectiveness |
| `victim_clamped_baseline` | None | Clamped | Cost of clamping on the victim alone |
| `recovery_baseline` | None | Restored to original | Confirms GPU returns to normal |

**Mitigation decomposition (the key T3 analysis):**
```
Net mitigation benefit  = (unmitigated latency) - (mitigated latency)
Clamping overhead       = (clamped baseline latency) - (recovery baseline latency)
True attack reduction   = Net benefit - Clamping overhead
```

**Output:** `T3/closed_loop_mitigation_v2.csv`

**Custom power cap (default is 70% of current limit):**
```bash
sudo python T3/closed_loop_mitigator_v2.py --duration 15 --cap 21
```

**Multiple trials for statistical rigor:**
```bash
sudo python T3/closed_loop_mitigator_v2.py --duration 15 --trials 5
```

---

### Alternative: Run Everything via Master Suite

```bash
sudo python gaia_master_suite_v2.py --thrust all --duration 15
```

Runs T1 → T2 → T3 in sequence. Requires `sudo` for T3. You can also run individual thrusts:

```bash
python gaia_master_suite_v2.py --thrust 1 --duration 15          # T1 only
python gaia_master_suite_v2.py --thrust 2                        # T2 only (needs T1 CSV)
sudo python gaia_master_suite_v2.py --thrust 3 --duration 15     # T3 only (needs T1 binary)
```

---

## Output Files to Collect

After all thrusts complete, these are the files that feed your paper and deliverables:

| File | Deliverable | Contents |
|---|---|---|
| `T1/controlled_3phase_experiment_v2.csv` | D1 | Per-sample telemetry across all trials and phases |
| `T1/power_sag_analysis_v2.png` | D1, D4 | 4-panel IEEE figure (power, voltage, di/dt, PSD) |
| `T2/gpu_defense_diagnostics_v2.png` | D2, D4 | 3-panel detection diagnostic (Kalman, spectrogram, PCA) |
| `T3/closed_loop_mitigation_v2.csv` | D3 | Per-sample mitigation data with anomaly/clamp flags |

---

## Metrics Checklist (from Proposal)

Use this to confirm you have collected everything the proposal promises:

### Task 1 Metrics (Attack Effectiveness)
- [ ] Power variance between phases (compute from `power_w` column std per phase)
- [ ] Peak power draw vs TDP and 120% oversubscription limit
- [ ] Victim latency degradation (`victim_total_latency_ms` pulsed vs benign vs alone)
- [ ] Induced throttling rate (`sw_power_cap`, `hw_slowdown` columns)
- [ ] Voltage sag depth (`voltage_v` minimum during attack)
- [ ] di/dt magnitude during attack vs baseline
- [ ] Paired t-test p-value for pulsed vs matched-benign (< 0.05 required)
- [ ] Cohen's d effect size

### Task 2 Metrics (Detection Effectiveness)
- [ ] Kalman-smoothed power trace showing attack envelope
- [ ] STFT spectrogram showing spectral energy redistribution during attack
- [ ] PCA cluster separation between benign and attack windows
- [ ] Anomaly detector classification accuracy (SAFE vs RESONANT ATTACK)
- [ ] Detection latency (~64 samples / ~0.6s at 100 Hz pynvml rate)

### Task 3 Metrics (Mitigation Effectiveness)
- [ ] Net mitigation benefit (unmitigated - mitigated latency)
- [ ] Clamping overhead (clamped baseline - recovery baseline latency)
- [ ] True attack reduction (net benefit - clamping overhead)
- [ ] Power capping effectiveness (mean power under clamp vs uncapped)
- [ ] Throttle rate reduction under mitigation
- [ ] Voltage improvement under mitigation (higher V_min)
- [ ] Recovery confirmation (recovery baseline matches victim_alone from T1)

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: No hardware telemetry source available` | GPU not visible or drivers not loaded | Run `nvidia-smi` manually |
| `RuntimeError: Cannot detect GPU compute capability` | `nvidia-smi` not on PATH or hangs on Jetson | The script tries `nvidia-smi` (5s timeout) then falls back to `torch.cuda.get_device_capability()`. Ensure either the CUDA driver or PyTorch CUDA is installed |
| `RuntimeError: nvcc compilation failed` | CUDA toolkit version mismatch | Check `nvcc --version` matches driver |
| `RuntimeError: PyTorch CUDA is not available` | PyTorch CPU-only wheel | Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `RuntimeError: pynvml not available` | Missing package | `pip install nvidia-ml-py` |
| `Failed to set power limit` (T3) | Not running as root | Use `sudo` |
| Power readings off by ~1000x on Jetson | INA3221 driver unit mismatch (uW vs mW) | Check `cat /sys/class/hwmon/hwmon*/power1_input` and adjust divisor |
| `victim_pulsed_stress` latency not elevated | Short duration or stress process died | Use `--duration 30`, check PID in output |

---

## What This Code Does NOT Yet Cover

The proposal describes some capabilities not yet implemented in these scripts. These are future work items:

| Proposal Item | Status | Notes |
|---|---|---|
| MPS/MIG multi-tenancy emulation | Not implemented | Proposal mentions running victim and attacker via NVIDIA MPS or MIG partitioning |
| Temperature-based attack synchronization | Not implemented | Proposal Section 3 describes using GPU temperature correlation as an attack timing cue |
| DCGM-based detection | Not implemented | Proposal Task 2 mentions NVIDIA Data Center GPU Manager as a monitoring tool |
| DCGM Exporter + Prometheus + Grafana stack | Not implemented | Proposal Task 3 describes an industry-standard monitoring pipeline prototype |
| Adversarial kernel termination | Not implemented | Proposal Task 3 mentions the ability to terminate adversarial kernels gracefully |
| Performance counter profiling | Not implemented | Proposal Task 2 mentions cache-line and performance counter monitoring |
