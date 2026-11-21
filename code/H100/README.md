# GAIA H100 Discrete GPU Validation

Scripts to validate that GAIA attack primitives transfer from the Jetson AGX Thor
to discrete data-center GPUs (H100/A100).

## Prerequisites

- NVIDIA H100 or A100 GPU with CUDA toolkit (nvcc) installed
- Python 3.8+ with numpy, scipy, pandas, scikit-learn
- PyTorch with CUDA support (for matmul victim)
- pynvml (`pip install nvidia-ml-py`) — recommended
- DCGM (`nv-hostengine` + `dcgmi`) — optional, for highest-frequency telemetry
- Root/sudo access for `nvidia-smi -pl` and MIG operations

## Scripts

### T1: Attack Transfer Validation
```bash
python h100_attack_transfer.py --trials 5 --duration 15
```
Runs 5 attack primitives against a matmul victim on the H100 and outputs CSV
compatible with the existing analysis pipeline.

### T1: Power-Cap Oversubscription Sweep
```bash
sudo python h100_powercap_sweep.py --trials 3 --caps 300,400,500,600,700
```
Sweeps nvidia-smi power limits to test whether software capping neutralizes attacks.

### T1: MIG Isolation Test
```bash
sudo python h100_mig_isolation.py --trials 5 --duration 20
```
Creates two MIG instances, runs victim in one and attacker in the other.
Tests whether power-domain interference crosses MIG boundaries.

### T2: DCGM-Based Detection
```bash
# Real-time monitor
python h100_dcgm_detector.py --mode monitor --duration 60

# Train classifier on collected data
python h100_dcgm_detector.py --mode classify --csv h100_transfer_results.csv
```
8-feature Random Forest classifier on DCGM/pynvml telemetry.

### T3: Closed-Loop Pipeline
```bash
sudo python h100_closed_loop_pipeline.py --trials 5 --duration 15
```
Full detect → mitigate → attribute → terminate pipeline on discrete GPU.

## Output

Each script creates a timestamped results directory with:
- CSV data compatible with `code/analyze_results.py`
- `platform_info.json` with full reproducibility metadata
- Console summary with key metrics
