from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    print(f"[run_pipeline] Running {script_path.name}...")
    subprocess.run([sys.executable, str(script_path)], check=True)
    print(f"[run_pipeline] Finished {script_path.name}")


def main() -> None:
    scripts = [
        "generate_data.py",
        "train_model.py",
        "optimal_control.py",
    ]
    for script in scripts:
        run_script(script)


if __name__ == "__main__":
    main()
