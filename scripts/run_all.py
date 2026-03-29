"""
Master runner: executes all experiments sequentially.
Each experiment saves results independently, so if anything crashes
we don't lose earlier work.
"""

import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ("00_setup", "experiment_00_setup.py"),
    ("01_activation_patching", "experiment_01_activation_patching.py"),
    ("02_logit_lens", "experiment_02_logit_lens.py"),
    ("03_entanglement", "experiment_03_entanglement.py"),
    ("04_cross_bias", "experiment_04_cross_bias.py"),
]

def run_experiment(name, script):
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, f"scripts/{script}"],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {name} completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ {name} FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        print("  Continuing to next experiment...")

    return result.returncode == 0


if __name__ == "__main__":
    print("="*70)
    print("MECHANISTIC INTERPRETABILITY FOR BIAS LOCALIZATION")
    print("Full Experiment Pipeline")
    print("="*70)

    total_start = time.time()
    results = {}

    for name, script in EXPERIMENTS:
        success = run_experiment(name, script)
        results[name] = success

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed:.1f}s total)")
    print(f"{'='*70}")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
