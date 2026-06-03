"""Seed initial data: run training and save reference baseline."""

import sys
sys.path.insert(0, ".")

from pipelines.training_pipeline import run_training_pipeline


def main():
    print("Seeding project with initial model and reference data...")
    result = run_training_pipeline()
    print(f"\nSeed complete!")
    print(f"  Model: {result['best_model']}")
    print(f"  AUPRC: {result['metrics']['auprc']:.4f}")
    print(f"  Path:  {result['model_path']}")


if __name__ == "__main__":
    main()
