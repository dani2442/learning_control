"""Optimise open-loop controls on real, learned, or both dynamics.

Modes
-----
real     – optimise on the real vortex dynamics.
learned  – optimise on the learned Neural SDE.
compare  – optimise on both and print a cross-evaluation table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control import (
    OptimalControlConfig,
    optimize_open_loop_controls,
    rollout,
    terminal_error,
)
from src.dataset import (
    add_control_type_to_path,
    config_from_payload,
    control_type_from_payload,
    load_dataset,
)
from src.dynamics import VortexSDE
from src.model import load_checkpoint
from src.visualization import plot_control_comparison, plot_single_control_solution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimise open-loop control sequences.")
    p.add_argument("--mode", choices=("real", "learned", "compare"), default="compare")
    p.add_argument("--dataset", type=str, default="data/controlled_vortex_constant.pt")
    p.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    p.add_argument("--x0", type=float, nargs=2, default=(-3.0, 1.2))
    p.add_argument("--target", type=float, nargs=2, default=(2.5, -0.8))
    p.add_argument("--horizon-steps", type=int, default=OptimalControlConfig.horizon_steps)
    p.add_argument("--iters", type=int, default=OptimalControlConfig.steps)
    p.add_argument("--max-abs-control", type=float, default=OptimalControlConfig.max_abs_control)
    p.add_argument("--control-lr", type=float, default=OptimalControlConfig.lr)
    p.add_argument("--terminal-weight", type=float, default=OptimalControlConfig.terminal_weight)
    p.add_argument("--effort-weight", type=float, default=OptimalControlConfig.effort_weight)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--plot", dest="plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-plot", dest="show_plot", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--force-step", type=int, default=3)
    p.add_argument("--image-dir", type=str, default="images")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(args, dt: float) -> OptimalControlConfig:
    return OptimalControlConfig(
        horizon_steps=args.horizon_steps,
        dt=dt,
        max_abs_control=args.max_abs_control,
        steps=args.iters,
        lr=args.control_lr,
        terminal_weight=args.terminal_weight,
        effort_weight=args.effort_weight,
    )


def _load_model(checkpoint: str, control_type: str, device: str):
    path = add_control_type_to_path(checkpoint, control_type)
    model, _ = load_checkpoint(path, device=device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _run_single(args, sde, x0, target, cfg, system_config, label: str):
    controls, traj, _ = optimize_open_loop_controls(sde, x0, target, cfg)
    err = terminal_error(traj, target)
    print(f"dynamics={label} terminal_error={err:.4f}")
    print(f"control_l2={float(torch.mean(controls.square()).sqrt().item()):.4f}")

    if args.plot:
        control_type = control_type_from_payload(load_dataset(args.dataset))
        force_gain = (float(system_config.control_gain_x), float(system_config.control_gain_y))
        plot_path = add_control_type_to_path(
            Path(args.image_dir) / f"compute_control_{label}.pdf",
            control_type, as_subdir=True,
        )
        plot_single_control_solution(
            trajectory=traj.cpu(),
            controls=controls.cpu(),
            target=target.cpu(),
            initial=x0.cpu(),
            force_gain=force_gain,
            config=system_config,
            title=f"Optimal control on {label} dynamics",
            force_step=args.force_step,
            save_path=plot_path,
            show=args.show_plot,
        )

    return controls, traj


def _run_compare(args, vortex_sde, model, x0, target, cfg, system_config):
    controls_real, traj_rr, _ = optimize_open_loop_controls(vortex_sde, x0, target, cfg)
    controls_learned, traj_ll, _ = optimize_open_loop_controls(model, x0, target, cfg)

    traj_rl = rollout(model, x0, controls_real, cfg.dt)
    traj_lr = rollout(vortex_sde, x0, controls_learned, cfg.dt)

    print("Comparison (terminal L2 error):")
    print(f"  optimize real,    eval real:    {terminal_error(traj_rr, target):.4f}")
    print(f"  optimize real,    eval learned: {terminal_error(traj_rl, target):.4f}")
    print(f"  optimize learned, eval learned: {terminal_error(traj_ll, target):.4f}")
    print(f"  optimize learned, eval real:    {terminal_error(traj_lr, target):.4f}")

    if args.plot:
        control_type = control_type_from_payload(load_dataset(args.dataset))
        force_gain = (float(system_config.control_gain_x), float(system_config.control_gain_y))
        plot_path = add_control_type_to_path(
            Path(args.image_dir) / "optimal_control_comparison.pdf",
            control_type, as_subdir=True,
        )
        plot_control_comparison(
            true_opt_traj=traj_rr.cpu(),
            learned_opt_traj=traj_lr.cpu(),
            target=target.cpu(),
            initial=x0.cpu(),
            controls_true=controls_real.cpu(),
            controls_learned=controls_learned.cpu(),
            force_gain=force_gain,
            force_step=args.force_step,
            config=system_config,
            save_path=plot_path,
            show=args.show_plot,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = args.device

    payload = load_dataset(args.dataset)
    control_type = control_type_from_payload(payload)
    system_config = config_from_payload(payload)
    dt = float(payload["times"][1] - payload["times"][0])
    cfg = _build_config(args, dt)

    x0 = torch.tensor(args.x0, dtype=torch.float32, device=device)
    target = torch.tensor(args.target, dtype=torch.float32, device=device)

    vortex_sde = VortexSDE(system_config)
    model = None

    if args.mode in ("learned", "compare"):
        model = _load_model(args.checkpoint, control_type, device)

    if args.mode == "real":
        _run_single(args, vortex_sde, x0, target, cfg, system_config, "real")
    elif args.mode == "learned":
        _run_single(args, model, x0, target, cfg, system_config, "learned")
    else:
        _run_compare(args, vortex_sde, model, x0, target, cfg, system_config)


if __name__ == "__main__":
    main()
