"""Run the full pipeline: generate data -> train model -> optimise control."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import add_control_type_to_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run data generation, model training, and control comparison.")
    p.add_argument("--control-type", choices=("constant", "sinusoidal"), default="constant")
    p.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    p.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    p.add_argument("--image-dir", type=str, default="images")
    return p.parse_args()


def _run(script: str, *script_args: str) -> None:
    path = Path(__file__).resolve().parent / script
    cmd = [sys.executable, str(path), *script_args]
    print(f"[pipeline] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    dataset_path = add_control_type_to_path(args.dataset, args.control_type)

    _run(
        "generate_data.py",
        "--control-type", args.control_type,
        "--output", args.dataset,
        "--image-dir", args.image_dir,
    )
    _run(
        "train_model.py",
        "--dataset", str(dataset_path),
        "--checkpoint", args.checkpoint,
        "--image-dir", args.image_dir,
    )
    _run(
        "optimize.py",
        "--mode", "compare",
        "--dataset", str(dataset_path),
        "--checkpoint", args.checkpoint,
        "--image-dir", args.image_dir,
    )


if __name__ == "__main__":
    main()
