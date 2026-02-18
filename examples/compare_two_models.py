"""End-to-end comparison example for two datasets and two learned models.

This script:
1. Generates two datasets (defaults: constant and sinusoidal controls).
2. Trains one Neural SDE model per dataset.
3. Plots a 2x2 figure: dataset trajectories (top) + vector-field error maps (bottom).
4. Solves optimal control on real dynamics and both learned models.
5. Plots a 3-column control figure with per-model optima and transfer to real dynamics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control import OptimalControlConfig, optimize_open_loop_controls, rollout, terminal_error
from src.controls import (
    RandomSinusoidalControlConfig,
    make_constant_control_fn,
    make_random_sinusoidal_control_fn,
)
from src.dataset import (
    ControlledTrajectoryDataset,
    add_control_type_to_path,
    config_from_payload,
    generate_dataset,
    save_dataset,
)
from src.dynamics import VortexSystemConfig, VortexSDE, controlled_vortex_drift, vortex_vector_field
from src.model import ControlledNeuralSDE, TrainingConfig, save_checkpoint, train_neural_sde


STREAM_DENSITY = 1.45
STREAM_LINE_WIDTH = 0.6
STREAM_ARROW_SIZE = 0.9
STREAM_COLOR = "#9aa6b2"
FORCE_SCALE_NORM = 0.7
QUIVER_WIDTH = 0.005
POLE_MASK_RADIUS_SQ = 0.15 ** 2
SPEED_CAP = 6.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/train/compare two controlled-dynamics models.")

    p.add_argument("--dataset1-type", choices=("constant", "sinusoidal"), default="constant")
    p.add_argument("--dataset2-type", choices=("constant", "sinusoidal"), default="sinusoidal")
    p.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    p.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")

    p.add_argument("--num-trajectories", type=int, default=512)
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--constant-values", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--sin-amplitude", type=float, default=0.6)
    p.add_argument("--sin-freq-range", type=float, nargs=2, default=(0.4, 2.0))
    p.add_argument("--sin-phase-range", type=float, nargs=2, default=(0.0, 2.0 * math.pi))
    p.add_argument("--sin-bias", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--x0", type=float, nargs=2, default=(-3.0, 1.2))
    p.add_argument("--target", type=float, nargs=2, default=(2.5, -0.8))
    p.add_argument("--horizon-steps", type=int, default=OptimalControlConfig.horizon_steps)
    p.add_argument("--iters", type=int, default=OptimalControlConfig.steps)
    p.add_argument("--max-abs-control", type=float, default=OptimalControlConfig.max_abs_control)
    p.add_argument("--control-lr", type=float, default=OptimalControlConfig.lr)
    p.add_argument("--terminal-weight", type=float, default=OptimalControlConfig.terminal_weight)
    p.add_argument("--effort-weight", type=float, default=OptimalControlConfig.effort_weight)
    p.add_argument("--force-step", type=int, default=3)

    p.add_argument("--max-lines", type=int, default=60)
    p.add_argument("--stream-grid", type=int, default=220)
    p.add_argument("--error-grid", type=int, default=150)
    p.add_argument("--error-vmax", type=float, default=5)
    p.add_argument("--image-dir", type=str, default="images")
    p.add_argument("--show-plot", dest="show_plot", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


def _build_control_fn(args: argparse.Namespace, control_type: str, seed: int):
    if control_type == "constant":
        return make_constant_control_fn(
            num_trajectories=args.num_trajectories,
            value=args.constant_values,
            seed=seed,
        )

    sin_cfg = RandomSinusoidalControlConfig(
        amplitude=args.sin_amplitude,
        frequency_range=(args.sin_freq_range[0], args.sin_freq_range[1]),
        phase_range=(args.sin_phase_range[0], args.sin_phase_range[1]),
        bias=args.sin_bias,
    )
    return make_random_sinusoidal_control_fn(
        num_trajectories=args.num_trajectories,
        config=sin_cfg,
        seed=seed,
    )


def _generate_one_dataset(
    args: argparse.Namespace,
    *,
    control_type: str,
    seed: int,
    config: VortexSystemConfig,
) -> tuple[dict[str, object], Path]:
    control_fn, control_meta = _build_control_fn(args, control_type, seed)
    payload = generate_dataset(
        num_trajectories=args.num_trajectories,
        horizon=args.horizon,
        dt=args.dt,
        seed=seed,
        config=config,
        control_fn=control_fn,
    )
    payload["control"] = control_meta

    out_path = add_control_type_to_path(args.dataset, control_type)
    save_dataset(payload, out_path)
    return payload, out_path


def _split_loaders(
    payload: dict[str, object],
    *,
    batch_size: int,
    val_ratio: float,
    split_seed: int,
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]] | None]:
    dataset = ControlledTrajectoryDataset(payload)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val-ratio must satisfy 0 <= val_ratio < 1, got {val_ratio}.")

    val_size = max(1, round(len(dataset) * val_ratio)) if val_ratio > 0.0 else 0
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_size > 0 else None
    return train_loader, val_loader


def _train_one_model(
    args: argparse.Namespace,
    payload: dict[str, object],
    control_type: str,
) -> tuple[ControlledNeuralSDE, dict[str, list[float]], Path]:
    train_loader, val_loader = _split_loaders(
        payload,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    model = ControlledNeuralSDE(hidden_dim=args.hidden_dim)
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        solver_dt=float(payload["times"][1] - payload["times"][0]),
    )
    history = train_neural_sde(
        model=model,
        dataloader=train_loader,
        times=payload["times"],
        config=train_cfg,
        val_dataloader=val_loader,
        device=args.device,
    )
    model.eval()

    checkpoint_path = add_control_type_to_path(args.checkpoint, control_type)
    save_checkpoint(model, checkpoint_path, train_cfg, history)
    return model, history, checkpoint_path


def _add_stream(ax: plt.Axes, config: VortexSystemConfig, xlim: tuple[float, float], ylim: tuple[float, float], grid_size: int) -> None:
    x_grid, y_grid, u_field, v_field, mask = vortex_vector_field(
        config=config,
        xlim=xlim,
        ylim=ylim,
        grid_size=grid_size,
        control_u=0.0,
    )
    ax.streamplot(
        x_grid,
        y_grid,
        np.ma.array(u_field, mask=mask),
        np.ma.array(v_field, mask=mask),
        density=STREAM_DENSITY,
        linewidth=STREAM_LINE_WIDTH,
        arrowsize=STREAM_ARROW_SIZE,
        color=STREAM_COLOR,
    )
    ax.scatter(config.poles, np.zeros(len(config.poles)), s=180, facecolors="none", edgecolors="black", linewidths=1.4)


@torch.no_grad()
def _add_model_stream(
    ax: plt.Axes,
    model: ControlledNeuralSDE,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
) -> None:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    model_device = next(model.parameters()).device
    points = torch.tensor(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1),
        dtype=torch.float32,
        device=model_device,
    )
    u = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
    t = torch.tensor(0.0, dtype=points.dtype, device=points.device)
    drift = model.drift_with_control(t=t, y=points, u=u).detach().cpu().numpy()

    u_field = drift[:, 0].reshape(grid_size, grid_size)
    v_field = drift[:, 1].reshape(grid_size, grid_size)

    speed = np.hypot(u_field, v_field)
    scale = np.minimum(1.0, SPEED_CAP / (speed + 1e-12))
    u_field *= scale
    v_field *= scale

    mask = np.zeros_like(x_grid, dtype=bool)
    for pole_x in config.poles:
        mask |= (x_grid - pole_x) ** 2 + y_grid ** 2 < POLE_MASK_RADIUS_SQ

    ax.streamplot(
        x_grid,
        y_grid,
        np.ma.array(u_field, mask=mask),
        np.ma.array(v_field, mask=mask),
        density=STREAM_DENSITY,
        linewidth=STREAM_LINE_WIDTH,
        arrowsize=STREAM_ARROW_SIZE,
        color=STREAM_COLOR,
    )
    ax.scatter(config.poles, np.zeros(len(config.poles)), s=180, facecolors="none", edgecolors="black", linewidths=1.4)


def _trajectory_bounds(states: Tensor, config: VortexSystemConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    arr = states.detach().cpu().numpy()
    x_vals = arr[:, :, 0]
    y_vals = arr[:, :, 1]
    xlim = (
        float(min(np.min(x_vals), min(config.poles)) - 1.0),
        float(max(np.max(x_vals), max(config.poles)) + 1.0),
    )
    ylim = (float(np.min(y_vals) - 1.0), float(np.max(y_vals) + 1.0))
    return xlim, ylim


@torch.no_grad()
def _compute_error_map(
    model: ControlledNeuralSDE,
    config: VortexSystemConfig,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    device: str,
) -> np.ndarray:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    points = torch.tensor(np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1), dtype=torch.float32, device=device)
    u = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=device)
    t = torch.tensor(0.0, dtype=points.dtype, device=device)

    true_drift = controlled_vortex_drift(points, u, config)
    learned_drift = model.drift_with_control(t=t, y=points, u=u)
    error = torch.linalg.vector_norm(learned_drift - true_drift, ord=2, dim=-1)
    return error.reshape(grid_size, grid_size).detach().cpu().numpy()


def plot_dataset_and_error_grid(
    payload_1: dict[str, object],
    payload_2: dict[str, object],
    model_1: ControlledNeuralSDE,
    model_2: ControlledNeuralSDE,
    *,
    label_1: str,
    label_2: str,
    stream_grid: int,
    error_grid: int,
    error_vmax: float | None,
    max_lines: int,
    device: str,
    save_path: str | Path,
    show: bool,
) -> None:
    states_1: Tensor = payload_1["states"]  # type: ignore[assignment]
    states_2: Tensor = payload_2["states"]  # type: ignore[assignment]
    cfg_1 = config_from_payload(payload_1)
    cfg_2 = config_from_payload(payload_2)

    # xlim_1, ylim_1 = _trajectory_bounds(states_1, cfg_1)
    # xlim_2, ylim_2 = _trajectory_bounds(states_2, cfg_2)
    # xlim = (min(xlim_1[0], xlim_2[0]), max(xlim_1[1], xlim_2[1]))
    # ylim = (min(ylim_1[0], ylim_2[0]), max(ylim_1[1], ylim_2[1]))
    xlim = (-4.0, 6.0)
    ylim = (-3.0, 3.0)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.06))
    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])
    cax = fig.add_subplot(gs[:, 2])

    for col, (ax, states, cfg, xlim, ylim, label) in enumerate([
        (axes[0, 0], states_1, cfg_1, xlim, ylim, label_1),
        (axes[0, 1], states_2, cfg_2, xlim, ylim, label_2),
    ]):
        _add_stream(ax, cfg, xlim, ylim, stream_grid)
        arr = states.detach().cpu().numpy()
        n = min(max_lines, arr.shape[0])
        shades = plt.cm.Blues(np.linspace(0.35, 0.95, n))
        for idx in range(n):
            ax.plot(arr[idx, :, 0], arr[idx, :, 1], color=shades[idx], lw=1.7, alpha=0.95)
        ax.set_title(f"Dataset {col + 1} ({label})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    error_map_1 = _compute_error_map(
        model_1,
        cfg_1,
        xlim=xlim,
        ylim=ylim,
        grid_size=error_grid,
        device=device,
    )
    error_map_2 = _compute_error_map(
        model_2,
        cfg_2,
        xlim=xlim,
        ylim=ylim,
        grid_size=error_grid,
        device=device,
    )
    common_vmin = float(min(np.min(error_map_1), np.min(error_map_2)))
    common_vmax = float(max(np.max(error_map_1), np.max(error_map_2))) if error_vmax is None else float(error_vmax)
    if common_vmax <= common_vmin:
        common_vmax = common_vmin + 1e-6

    color_ref = None
    for ax, error_map, cfg, label in [
        (axes[1, 0], error_map_1, cfg_1, label_1),
        (axes[1, 1], error_map_2, cfg_2, label_2),
    ]:
        img = ax.imshow(
            error_map,
            origin="lower",
            extent=(*xlim, *ylim),
            aspect="equal",
            cmap="magma",
            vmin=common_vmin,
            vmax=common_vmax,
        )
        color_ref = img
        ax.scatter(cfg.poles, np.zeros(len(cfg.poles)), s=120, facecolors="none", edgecolors="white", linewidths=1.2)
        ax.set_title(f"Error Map ({label})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    if color_ref is not None:
        cbar = fig.colorbar(color_ref, cax=cax)
        cbar.set_label("L2 drift error")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    if show:
        plt.show()
    plt.close(fig)


def _control_plot_bounds(
    config: VortexSystemConfig,
    trajectories: list[Tensor],
    x0: Tensor,
    target: Tensor,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [traj[:, 0].detach().cpu().numpy() for traj in trajectories]
    ys = [traj[:, 1].detach().cpu().numpy() for traj in trajectories]

    all_x = np.concatenate([*xs, np.array([x0[0].item(), target[0].item(), *config.poles], dtype=float)])
    all_y = np.concatenate([*ys, np.array([x0[1].item(), target[1].item()], dtype=float)])
    xlim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)
    ylim = (float(all_y.min()) - 1.0, float(all_y.max()) + 1.0)
    return xlim, ylim


def _force_arrows(
    traj_np: np.ndarray,
    controls: Tensor,
    force_gain: tuple[float, float],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    ctrl = controls.detach().cpu().numpy().ravel()
    usable = min(ctrl.shape[0], traj_np.shape[0] - 1)
    if usable <= 0:
        return np.empty((0, 2)), np.empty((0, 2))
    ctrl = ctrl[:usable]
    vecs = np.column_stack((force_gain[0] * ctrl, force_gain[1] * ctrl))
    return traj_np[:usable:stride], vecs[::stride]


def _normalise_forces(*vec_arrays: np.ndarray) -> None:
    max_norm = max(
        (float(np.max(np.linalg.norm(v, axis=1))) if v.size else 0.0)
        for v in vec_arrays
    )
    if max_norm > 1e-12:
        scale = FORCE_SCALE_NORM / max_norm
        for v in vec_arrays:
            v *= scale


def plot_control_three_columns(
    *,
    config: VortexSystemConfig,
    model_1: ControlledNeuralSDE,
    model_2: ControlledNeuralSDE,
    x0: Tensor,
    target: Tensor,
    controls_model_1: Tensor,
    controls_model_2: Tensor,
    traj_model_1: Tensor,
    traj_model_2: Tensor,
    traj_true: Tensor,
    traj_model_1_real: Tensor,
    traj_model_2_real: Tensor,
    label_1: str,
    label_2: str,
    force_step: int,
    stream_grid: int,
    save_path: str | Path,
    show: bool,
) -> None:
    xlim, ylim = _control_plot_bounds(
        config,
        [traj_model_1, traj_model_2, traj_true, traj_model_1_real, traj_model_2_real],
        x0,
        target,
    )
    x0_np = x0.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    traj_model_1_np = traj_model_1.detach().cpu().numpy()
    traj_model_2_np = traj_model_2.detach().cpu().numpy()
    stride = max(1, force_step)
    force_gain = (float(config.control_gain_x), float(config.control_gain_y))
    pts_1, vecs_1 = _force_arrows(traj_model_1_np, controls_model_1, force_gain, stride)
    pts_2, vecs_2 = _force_arrows(traj_model_2_np, controls_model_2, force_gain, stride)
    _normalise_forces(vecs_1, vecs_2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    panels = [
        (
            axes[0],
            [(traj_model_1, "#005f73", f"Optimal on {label_1}")],
            f"Model 1 Dynamics ({label_1})",
            (pts_1, vecs_1, "#0a9396"),
            "model_1",
        ),
        (
            axes[1],
            [(traj_model_2, "#bb3e03", f"Optimal on {label_2}")],
            f"Model 2 Dynamics ({label_2})",
            (pts_2, vecs_2, "#ae2012"),
            "model_2",
        ),
        (
            axes[2],
            [
                (traj_true, "#0a9396", "Optimal on real"),
                (traj_model_1_real, "#005f73", f"{label_1} optimal, eval real"),
                (traj_model_2_real, "#bb3e03", f"{label_2} optimal, eval real"),
            ],
            "Transfer to Real Dynamics",
            None,
            "real",
        ),
    ]

    for ax, curves, title, force_data, stream_source in panels:
        if stream_source == "model_1":
            _add_model_stream(ax, model_1, config, xlim, ylim, stream_grid)
        elif stream_source == "model_2":
            _add_model_stream(ax, model_2, config, xlim, ylim, stream_grid)
        else:
            _add_stream(ax, config, xlim, ylim, stream_grid)
        for traj, color, label in curves:
            arr = traj.detach().cpu().numpy()
            ax.plot(arr[:, 0], arr[:, 1], lw=2.2, color=color, label=label)
        if force_data is not None:
            pts, vecs, force_color = force_data
            if pts.shape[0] > 0:
                ax.quiver(
                    pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
                    color=force_color, alpha=0.95, angles="xy",
                    scale_units="xy", scale=1.0, width=QUIVER_WIDTH,
                    label="Control force (scaled)",
                )
        ax.scatter(*x0_np, s=80, marker="o", color="#1f2933", label="Initial")
        ax.scatter(*target_np, s=140, marker="*", color="#ee9b00", label="Target")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = VortexSystemConfig()
    label_1 = args.dataset1_type
    label_2 = args.dataset2_type

    payload_1, dataset_path_1 = _generate_one_dataset(
        args,
        control_type=label_1,
        seed=args.seed,
        config=config,
    )
    payload_2, dataset_path_2 = _generate_one_dataset(
        args,
        control_type=label_2,
        seed=args.seed + 1,
        config=config,
    )
    print(f"dataset_1={dataset_path_1}")
    print(f"dataset_2={dataset_path_2}")

    model_1, history_1, checkpoint_1 = _train_one_model(args, payload_1, label_1)
    model_2, history_2, checkpoint_2 = _train_one_model(args, payload_2, label_2)
    print(f"checkpoint_1={checkpoint_1} final_train_loss={history_1['train_loss'][-1]:.6f}")
    print(f"checkpoint_2={checkpoint_2} final_train_loss={history_2['train_loss'][-1]:.6f}")

    grid_path = Path(args.image_dir) / "two_datasets_two_error_maps.pdf"
    plot_dataset_and_error_grid(
        payload_1,
        payload_2,
        model_1,
        model_2,
        label_1=label_1,
        label_2=label_2,
        stream_grid=args.stream_grid,
        error_grid=args.error_grid,
        error_vmax=args.error_vmax,
        max_lines=args.max_lines,
        device=args.device,
        save_path=grid_path,
        show=args.show_plot,
    )
    print(f"saved={grid_path}")

    dt_1 = float(payload_1["times"][1] - payload_1["times"][0])
    dt_2 = float(payload_2["times"][1] - payload_2["times"][0])
    if abs(dt_1 - dt_2) > 1e-9:
        raise ValueError(f"Datasets must share the same dt for fair control comparison, got dt1={dt_1}, dt2={dt_2}.")

    control_cfg = OptimalControlConfig(
        horizon_steps=args.horizon_steps,
        dt=dt_1,
        max_abs_control=args.max_abs_control,
        steps=args.iters,
        lr=args.control_lr,
        terminal_weight=args.terminal_weight,
        effort_weight=args.effort_weight,
    )

    x0 = torch.tensor(args.x0, dtype=torch.float32, device=args.device)
    target = torch.tensor(args.target, dtype=torch.float32, device=args.device)
    real_sde = VortexSDE(config)

    _, traj_true, _ = optimize_open_loop_controls(real_sde, x0, target, control_cfg)
    controls_m1, traj_m1, _ = optimize_open_loop_controls(model_1, x0, target, control_cfg)
    controls_m2, traj_m2, _ = optimize_open_loop_controls(model_2, x0, target, control_cfg)

    traj_m1_real = rollout(real_sde, x0, controls_m1, control_cfg.dt)
    traj_m2_real = rollout(real_sde, x0, controls_m2, control_cfg.dt)

    print("Terminal L2 errors:")
    print(f"  optimize real, eval real: {terminal_error(traj_true, target):.4f}")
    print(f"  optimize {label_1}, eval {label_1}: {terminal_error(traj_m1, target):.4f}")
    print(f"  optimize {label_2}, eval {label_2}: {terminal_error(traj_m2, target):.4f}")
    print(f"  optimize {label_1}, eval real: {terminal_error(traj_m1_real, target):.4f}")
    print(f"  optimize {label_2}, eval real: {terminal_error(traj_m2_real, target):.4f}")

    control_path = Path(args.image_dir) / "three_way_control_comparison.pdf"
    plot_control_three_columns(
        config=config,
        model_1=model_1,
        model_2=model_2,
        x0=x0,
        target=target,
        controls_model_1=controls_m1,
        controls_model_2=controls_m2,
        traj_model_1=traj_m1,
        traj_model_2=traj_m2,
        traj_true=traj_true,
        traj_model_1_real=traj_m1_real,
        traj_model_2_real=traj_m2_real,
        label_1=label_1,
        label_2=label_2,
        force_step=args.force_step,
        stream_grid=args.stream_grid,
        save_path=control_path,
        show=args.show_plot,
    )
    print(f"saved={control_path}")


if __name__ == "__main__":
    main()
