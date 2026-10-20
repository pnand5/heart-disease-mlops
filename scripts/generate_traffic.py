"""Send mixed /predict traffic so the Grafana dashboard has data.

Usage: python scripts/generate_traffic.py [--url http://localhost:8000] [--seconds 60]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import FEATURES, load_clean  # noqa: E402


def payload(row) -> dict:
    return {c: float(row[c]) if c in ("age", "trestbps", "chol", "thalach", "oldpeak")
            else int(row[c]) for c in FEATURES}


def main(url: str, seconds: int) -> None:
    df = load_clean()
    rng = random.Random(7)
    end = time.time() + seconds
    sent = {"ok": 0, "invalid": 0, "failed": 0}
    while time.time() < end:
        roll = rng.random()
        try:
            if roll < 0.12:  # deliberately bad input -> 422
                bad = payload(df.iloc[rng.randrange(len(df))])
                bad["age"] = -5
                requests.post(f"{url}/predict", json=bad, timeout=5)
                sent["invalid"] += 1
            elif roll < 0.18:
                requests.get(f"{url}/health", timeout=5)
            else:
                row = df.iloc[rng.randrange(len(df))]
                requests.post(f"{url}/predict", json=payload(row), timeout=5)
                sent["ok"] += 1
        except requests.RequestException:
            sent["failed"] += 1
        time.sleep(rng.uniform(0.1, 0.6))
    print(f"done: {sent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    main(args.url, args.seconds)
