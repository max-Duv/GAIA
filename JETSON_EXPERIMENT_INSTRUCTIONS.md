# Jetson AGX Thor: Experiment Instructions

**Purpose:** Run these experiments on the Jetson to close remaining proposal gaps. Ordered by priority.

---

## Experiment 1: Filler Kernel Defense (T2 — proposal gap)

The proposal lists "filler kernels" as a defense mechanism. This experiment tests whether proactively filling idle GPU capacity with benign work dampens attack power amplitude.

### Copy files to Jetson
```bash
scp code/filler_kernel.cu code/filler_kernel_wrapper.py \
    T2/filler_kernel_experiment.py \
    code/T1/gpu_fault_assessor_v2.py \
    code/T1/gpu_power_poc_v3.cu \
    spire@jetson:~/CYSE-698-Tests/
```

### Compile on Jetson
```bash
ssh spire@jetson
cd ~/CYSE-698-Tests
nvcc -arch=sm_110 -O2 -o filler_kernel filler_kernel.cu
# Attacker binary should already exist; if not:
nvcc -arch=sm_110 -O2 -o gpu_power_poc_v3 gpu_power_poc_v3.cu -lcuda -lcudart
```

### Smoke test
```bash
./filler_kernel --intensity 50 --duty 0.5 --duration 5
# Should run 5s at ~50% SM occupancy, then exit
```

### Run experiment (5 trials, ~8 min total)
```bash
sudo python3 T2/filler_kernel_experiment.py \
    --trials 5 --duration 15 \
    --output-dir results/filler_kernel/
```

### Copy results back to Windows
```bash
scp -r spire@jetson:~/CYSE-698-Tests/results/filler_kernel/ \
    "C:\Users\maxfd\OneDrive\Documents\PhD\CYSE-698\GAIA-USENIX\11.06.2026_results\results\T2\"
```

---

## Experiment 2: LLM Workload Characterization (T1 — proposal gap)

The proposal commits to "model the phases of power consumption during LLM training and inference." This uses GPT-2 small (124M params).

### Install dependencies on Jetson
```bash
pip3 install transformers datasets accelerate
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('gpt2')"
```

### Copy files
```bash
scp T1/llm_workload_experiment.py \
    code/T1/gpu_fault_assessor_v2.py \
    spire@jetson:~/CYSE-698-Tests/
```

### Run LLM training experiment
```bash
sudo python3 T1/llm_workload_experiment.py \
    --victim-type training --model gpt2 \
    --batch-size 2 --seq-len 128 \
    --trials 5 --duration 15 \
    --out results/llm_training/llm_training.csv
```

### Run LLM inference experiment
```bash
sudo python3 T1/llm_workload_experiment.py \
    --victim-type inference --model gpt2 \
    --batch-size 4 --seq-len 64 \
    --trials 5 --duration 15 \
    --out results/llm_inference/llm_inference.csv
```

### If OOM
```bash
# Reduce: --batch-size 1 --seq-len 64 for training
# Reduce: --batch-size 2 --seq-len 32 for inference
```

### Copy results back
```bash
scp -r spire@jetson:~/CYSE-698-Tests/results/llm_*/ \
    "C:\Users\maxfd\OneDrive\Documents\PhD\CYSE-698\GAIA-USENIX\11.06.2026_results\results\T1\"
```

---

## Experiment 3: Prometheus Exporter Verification (T3 — proposal gap)

The proposal commits to "industry-standard monitoring tools." This verifies the Prometheus exporter works end-to-end.

### Install and run
```bash
pip3 install prometheus_client
scp code/prometheus_exporter.py spire@jetson:~/CYSE-698-Tests/
ssh spire@jetson
sudo python3 prometheus_exporter.py --port 9101 --threshold 60
```

### Verify metrics
```bash
curl http://localhost:9101/metrics | head -30
# Should show gaia_gpu_power_watts, gaia_gpu_clock_hz, etc.
```

### Screenshot the output for the paper
Take a screenshot of the /metrics output and/or Grafana dashboard if available.

---

## After All Experiments

Run analysis on Windows:
```bash
cd "C:\Users\maxfd\OneDrive\Documents\PhD\CYSE-698\GAIA-USENIX"

# LLM analysis (generates figures + LaTeX table)
python T1/llm_phase_analysis.py

# Then we integrate results into the paper
```
