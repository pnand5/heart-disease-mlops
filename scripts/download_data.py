"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/."""
import sys
from pathlib import Path

import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "processed.cleveland.data"


def download(url: str = URL, dest: Path = DEST, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"Already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"Saved {len(resp.content)} bytes to {dest}")
    return dest


if __name__ == "__main__":
    try:
        download(force="--force" in sys.argv)
    except requests.RequestException as exc:
        sys.exit(f"Download failed: {exc}")
