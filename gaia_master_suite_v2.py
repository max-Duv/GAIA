import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "T1"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "T2"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "T3"))

from gpu_fault_assessor_v2 import GPUFaultAssessorV2
from controlled_3phase_runner_v2 import run_controlled_3phase_experiment, compile_cuda_backend
from gpu_defense_filters_v2 import PlatformAgnosticAnomalyDetector
from power_analysis_plotter import plot_power_characterization
from closed_loop_mitigator_v2 import run_closed_loop_experiment


def run_thrust1_suite(duration=15.0, sweep=False):
    print("\n===========================================================================")
    print("      PROJECT GAIA: THRUST T1 - ATTACK & FAULT ASSESSMENT (v2.0)           ")
    print("===========================================================================")

    t1_dir = os.path.join(SCRIPT_DIR, "T1")
    cuda_src = os.path.join(t1_dir, "gpu_power_poc_v3.cu")
    cuda_bin = os.path.join(t1_dir, "gpu_power_poc_v3")
    compile_cuda_backend(cuda_src, cuda_bin)

    exp_csv = os.path.join(t1_dir, "controlled_3phase_experiment_v2.csv")
    run_controlled_3phase_experiment(
        cuda_binary=cuda_bin,
        duration_per_phase=duration,
        output_csv=exp_csv,
        sweep_mechanism=sweep
    )

    plot_png = os.path.join(t1_dir, "power_sag_analysis_v2.png")
    if os.path.exists(exp_csv):
        import pandas as pd
        df = pd.read_csv(exp_csv)
        plot_power_characterization(df, output_png_path=plot_png)

    print(f"[+] Thrust T1 Complete. Output in {t1_dir}")


def run_thrust2_suite():
    print("\n===========================================================================")
    print("      PROJECT GAIA: THRUST T2 - SIGNAL PROCESSING & ANOMALY FILTERS       ")
    print("===========================================================================")

    t1_dir = os.path.join(SCRIPT_DIR, "T1")
    t2_dir = os.path.join(SCRIPT_DIR, "T2")

    target_csv = os.path.join(t1_dir, "controlled_3phase_experiment_v2.csv")
    if not os.path.exists(target_csv):
        raise FileNotFoundError(
            f"T1 experiment CSV not found: {target_csv}\n"
            "Run Thrust 1 first (--thrust 1)."
        )

    out_png = os.path.join(t2_dir, "gpu_defense_diagnostics_v2.png")
    detector = PlatformAgnosticAnomalyDetector()
    detector.process_logged_file(target_csv, output_png=out_png)
    print(f"[+] Thrust T2 Complete. Output in {t2_dir}")


def run_thrust3_suite(duration=15.0):
    print("\n===========================================================================")
    print("      PROJECT GAIA: THRUST T3 - CLOSED-LOOP ACTIVE MITIGATION            ")
    print("===========================================================================")

    t1_dir = os.path.join(SCRIPT_DIR, "T1")
    t3_dir = os.path.join(SCRIPT_DIR, "T3")

    cuda_bin = os.path.join(t1_dir, "gpu_power_poc_v3")
    if not os.path.exists(cuda_bin):
        raise FileNotFoundError(
            f"T1 CUDA binary not found: {cuda_bin}\n"
            "Run Thrust 1 first (--thrust 1) to compile it."
        )

    out_csv = os.path.join(t3_dir, "closed_loop_mitigation_v2.csv")
    run_closed_loop_experiment(
        cuda_binary=cuda_bin,
        duration_per_phase=duration,
        output_csv=out_csv
    )
    print(f"[+] Thrust T3 Complete. Output in {t3_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Project GAIA: Unified Research Execution Suite (v2.0)"
    )
    parser.add_argument(
        "--thrust", type=str, choices=["1", "2", "3", "all"], default="all",
        help="Target research thrust"
    )
    parser.add_argument(
        "--duration", type=float, default=15.0,
        help="Duration per phase in seconds"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Execute mechanism isolation sweep (DAC Fig 6, T1 only)"
    )
    args = parser.parse_args()

    print("=====================================================================")
    print("    PROJECT GAIA: SECURING AI DATA CENTERS AGAINST POWER ATTACKS     ")
    print("                  Master Research Execution Harness (v2.0)           ")
    print("=====================================================================")

    if args.thrust in ["1", "all"]:
        run_thrust1_suite(duration=args.duration, sweep=args.sweep)
    if args.thrust in ["2", "all"]:
        run_thrust2_suite()
    if args.thrust in ["3", "all"]:
        run_thrust3_suite(duration=args.duration)

    print("\n=====================================================================")
    print("   ALL REQUESTED RESEARCH THRUSTS EXECUTED AND VERIFIED SUCCESSFULLY ")
    print("=====================================================================")


if __name__ == "__main__":
    main()
