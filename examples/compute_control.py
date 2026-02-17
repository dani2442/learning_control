from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
from src.dataset import config_from_payload, controlled_vortex_drift, load_dataset
from src.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute open-loop optimal control sequence.")
    parser.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    parser.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    parser.add_argument("--dynamics", choices=("real", "learned"), default="real")
    parser.add_argument("--x0", type=float, nargs=2, default=(-3.0, 1.2))
    parser.add_argument("--target", type=float, nargs=2, default=(2.5, -0.8))
    parser.add_argument("--horizon-steps", type=int, default=80)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)
    parser.add_argument("--field-grid", type=int, default=140)
    parser.add_argument("--force-step", type=int, default=3)
    parser.add_argument("--image-dir", type=str, default="images")
    return parser.parse_args()


@torch.no_grad()
def _build_vector_field(
    dynamics: str,
    system_config,
    model,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    grid_points = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    points = torch.tensor(grid_points, dtype=torch.float32, device=device)
    zeros_u = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=device)

    if dynamics == "real":
        drift = controlled_vortex_drift(points, zeros_u, system_config)
    else:
        t = torch.tensor(0.0, dtype=points.dtype, device=device)
        drift = model.drift_with_control(t=t, y=points, u=zeros_u)

    drift_np = drift.cpu().numpy()
    dx = drift_np[:, 0].reshape(grid_size, grid_size)
    dy = drift_np[:, 1].reshape(grid_size, grid_size)

    speed = np.hypot(dx, dy)
    cap = 4.5
    scale = np.minimum(1.0, cap / (speed + 1e-12))
    mask = np.zeros_like(x_grid, dtype=bool)
    for pole_x in system_config.poles:
        mask |= (x_grid - pole_x) ** 2 + y_grid**2 < 0.15**2
    return x_grid, y_grid, dx * scale, dy * scale, mask


def _plot_control_solution(
    dynamics: str,
    trajectory: torch.Tensor,
    controls: torch.Tensor,
    target: torch.Tensor,
    x0: torch.Tensor,
    force_gain: tuple[float, float],
    system_config,
    model,
    grid_size: int,
    force_step: int,
    device: str,
    save_path: str | Path,
) -> None:
    traj = trajectory.detach().cpu().numpy()
    control_vals = controls.detach().cpu().numpy().reshape(-1)
    poles = np.array(system_config.poles, dtype=float)

    x_margin = 1.0
    y_margin = 1.0
    x_min = float(min(traj[:, 0].min(), x0[0].item(), target[0].item(), poles.min()) - x_margin)
    x_max = float(max(traj[:, 0].max(), x0[0].item(), target[0].item(), poles.max()) + x_margin)
    y_min = float(min(traj[:, 1].min(), x0[1].item(), target[1].item()) - y_margin)
    y_max = float(max(traj[:, 1].max(), x0[1].item(), target[1].item()) + y_margin)

    x_grid, y_grid, vx, vy, field_mask = _build_vector_field(
        dynamics=dynamics,
        system_config=system_config,
        model=model,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        grid_size=grid_size,
        device=device,
    )

    force_vecs = np.column_stack((force_gain[0] * control_vals, force_gain[1] * control_vals))
    stride = max(1, force_step)
    force_points = traj[:-1:stride]
    force_vecs = force_vecs[::stride]
    force_norm = np.linalg.norm(force_vecs, axis=1)
    max_force = float(np.max(force_norm)) if force_norm.size else 0.0
    if max_force > 1e-12:
        force_vecs = force_vecs * (0.7 / max_force)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.streamplot(
        x_grid,
        y_grid,
        np.ma.array(vx, mask=field_mask),
        np.ma.array(vy, mask=field_mask),
        density=1.35,
        linewidth=0.65,
        arrowsize=0.9,
        color="#9aa6b2",
    )
    ax.plot(traj[:, 0], traj[:, 1], lw=2.2, color="#005f73", label="Optimal trajectory")
    ax.scatter(x0[0].item(), x0[1].item(), s=80, marker="o", color="#005f73", label="Initial")
    ax.scatter(target[0].item(), target[1].item(), s=140, marker="*", color="#ca6702", label="Target")
    ax.scatter(poles, np.zeros_like(poles), s=120, facecolors="none", edgecolors="black", linewidths=1.2)

    if force_points.shape[0] > 0:
        ax.quiver(
            force_points[:, 0],
            force_points[:, 1],
            force_vecs[:, 0],
            force_vecs[:, 1],
            color="#bb3e03",
            alpha=0.95,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            label="Control force (scaled)",
        )

    ax.set_title(f"Optimal control on {dynamics} dynamics")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    plt.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.show()


def main() -> None:
    args = parse_args()
    device = args.device

    payload = load_dataset(args.dataset)
    system_config = config_from_payload(payload)

    dt = float(payload["times"][1] - payload["times"][0])
    cfg = OptimalControlConfig(
        horizon_steps=args.horizon_steps,
        dt=dt,
        steps=args.iters,
    )

    x0 = torch.tensor(args.x0, dtype=torch.float32, device=device)
    target = torch.tensor(args.target, dtype=torch.float32, device=device)

    model = None
    if args.dynamics == "real":
        rollout_fn = partial(rollout_real_dynamics, system_config=system_config)
        force_gain = (system_config.control_gain_x, system_config.control_gain_y)
    else:
        model, _ = load_checkpoint(args.checkpoint, device=device)
        model.eval()
        rollout_fn = partial(rollout_learned_dynamics, model)
        force_gain = (float(model.control_matrix[0, 0].item()), float(model.control_matrix[1, 0].item()))

    controls, trajectory, _ = optimize_open_loop_controls(
        rollout_fn=rollout_fn,
        x0=x0,
        target=target,
        config=cfg,
        device=device,
    )

    err = terminal_error(trajectory, target)
    print(f"dynamics={args.dynamics} terminal_error={err:.4f}")
    print(f"control_l2={float(torch.mean(controls.square()).sqrt().item()):.4f}")

    if args.plot:
        _plot_control_solution(
            dynamics=args.dynamics,
            trajectory=trajectory,
            controls=controls,
            target=target,
            x0=x0,
            force_gain=force_gain,
            system_config=system_config,
            model=model,
            grid_size=args.field_grid,
            force_step=args.force_step,
            device=device,
            save_path=Path(args.image_dir) / f"compute_control_{args.dynamics}.pdf",
        )


if __name__ == "__main__":
    main()
