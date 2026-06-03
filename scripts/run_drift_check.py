"""Utility script to run a drift check from command line."""

import argparse
import sys
sys.path.insert(0, ".")

from pipelines.drift_pipeline import run_drift_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run drift detection check.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to current data CSV. If omitted, uses demo split."
    )
    args = parser.parse_args()

    result = run_drift_pipeline(current_data_path=args.data)
    print(f"\n{'='*50}")
    print(f"Drift Type:  {result['drift_type']}")
    print(f"Severity:    {result['severity']}")
    print(f"Action:      {result['action']}")
    print(f"Urgency:     {result['urgency_hours']} hours")
    print(f"Confidence:  {result['confidence']:.3f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
