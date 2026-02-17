from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control import (
    OptimalControlConfig,
    optimize_open_loop_controls,
    rollout_learned_dynamics,
    rollout_real_dynamics,
    terminal_error,
)
from src.dataset import config_from_payload, load_dataset
from src.model import load_checkpoint
from src.visualization import plot_control_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare optimal controls from real vs learned dynamics.")
    parser.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    parser.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    parser.add_argument("--x0", type=float, nargs=2, default=(-3.0, 1.2))
    parser.add_argument("--target", type=float, nargs=2, default=(2.5, -0.8))
    parser.add_argument("--horizon-steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", default=True, action="store_true")
    parser.add_argument("--force-step", type=int, default=3)
    parser.add_argument("--image-dir", type=str, default="images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    payload = load_dataset(args.dataset)
    system_config = config_from_payload(payload)
    dt = float(payload["times"][1] - payload["times"][0])

    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    x0 = torch.tensor(args.x0, dtype=torch.float32, device=device)
    target = torch.tensor(args.target, dtype=torch.float32, device=device)

    cfg = OptimalControlConfig(
        horizon_steps=args.horizon_steps,
        dt=dt,
        steps=args.iters,
    )
    real_force_gain = (float(system_config.control_gain_x), float(system_config.control_gain_y))

    rollout_real = partial(rollout_real_dynamics, system_config=system_config)
    rollout_learned = partial(rollout_learned_dynamics, model)

    controls_real_opt, traj_real_opt_on_real, _ = optimize_open_loop_controls(
        rollout_fn=rollout_real,
        x0=x0,
        target=target,
        config=cfg,
        device=device,
    )

    controls_learned_opt, traj_learned_opt_on_learned, _ = optimize_open_loop_controls(
        rollout_fn=rollout_learned,
        x0=x0,
        target=target,
        config=cfg,
        device=device,
    )

    traj_real_opt_on_learned = rollout_learned(x0, controls_real_opt, dt)
    traj_learned_opt_on_real = rollout_real(x0, controls_learned_opt, dt)

    print("Comparison (terminal L2 error):")
    print(f"- optimize on real, evaluate on real:      {terminal_error(traj_real_opt_on_real, target):.4f}")
    print(f"- optimize on real, evaluate on learned:   {terminal_error(traj_real_opt_on_learned, target):.4f}")
    print(f"- optimize on learned, evaluate on learned:{terminal_error(traj_learned_opt_on_learned, target):.4f}")
    print(f"- optimize on learned, evaluate on real:   {terminal_error(traj_learned_opt_on_real, target):.4f}")

    if args.plot:
        plot_control_comparison(
            true_opt_traj=traj_real_opt_on_real.cpu(),
            learned_opt_traj=traj_learned_opt_on_real.cpu(),
            target=target.cpu(),
            initial=x0.cpu(),
            controls_true=controls_real_opt.cpu(),
            controls_learned=controls_learned_opt.cpu(),
            force_gain=real_force_gain,
            force_step=args.force_step,
            config=system_config,
            save_path=Path(args.image_dir) / "optimal_control_comparison.pdf",
        )


if __name__ == "__main__":
    main()
