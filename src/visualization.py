"""Plotting utilities for trajectories, stream fields, and error maps."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from src.dataset import config_from_payload
from src.dynamics import (
    VORTEX_CONTROL_DIM,
    VortexSystemConfig,
    controlled_vortex_drift,
    vortex_vector_field,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STREAM_DENSITY = 1.45
STREAM_LINE_WIDTH = 0.6
STREAM_ARROW_SIZE = 0.9
STREAM_COLOR = "#9aa6b2"
FORCE_SCALE_NORM = 0.7
QUIVER_WIDTH = 0.005
POLE_MASK_RADIUS_SQ = 0.15**2
SPEED_CAP = 6.0


def _as_control_pair(control_u: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(control_u, tuple):
        if len(control_u) != VORTEX_CONTROL_DIM:
            raise ValueError(
                f"control_u tuple must have length {VORTEX_CONTROL_DIM}, got {len(control_u)}."
            )
        return float(control_u[0]), float(control_u[1])
    value = float(control_u)
    return value, value


def _control_tensor_for_dim(
    control_u: float | tuple[float, float],
    *,
    batch_size: int,
    control_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> Tensor:
    pair = _as_control_pair(control_u)
    if control_dim <= 0:
        raise ValueError(f"control_dim must be positive, got {control_dim}.")

    base = torch.zeros(control_dim, dtype=dtype, device=device)
    base[0] = pair[0]
    if control_dim >= 2:
        base[1] = pair[1]
    return base.unsqueeze(0).expand(batch_size, control_dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finalize_plot(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    *,
    show: bool = False,
    fmt: str = "pdf",
) -> None:
    """Save and/or show *fig*, then close it."""
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format=fmt)
    if show:
        plt.show()
    plt.close(fig)


def _add_stream_field(
    ax: plt.Axes,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 220,
    control_u: float | tuple[float, float] = 0.0,
) -> None:
    """Draw streamlines and pole markers on *ax*."""
    x_grid, y_grid, u_field, v_field, mask = vortex_vector_field(
        config, xlim, ylim, grid_size, control_u,
    )
    ax.streamplot(
        x_grid, y_grid,
        np.ma.array(u_field, mask=mask),
        np.ma.array(v_field, mask=mask),
        density=STREAM_DENSITY,
        linewidth=STREAM_LINE_WIDTH,
        arrowsize=STREAM_ARROW_SIZE,
        color=STREAM_COLOR,
    )
    ax.scatter(
        config.poles, np.zeros(len(config.poles)),
        s=180, facecolors="none", edgecolors="black", linewidths=1.4,
    )


@torch.no_grad()
def _add_model_stream_field(
    ax: plt.Axes,
    model,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 220,
) -> None:
    """Draw learned-model streamlines and pole markers on *ax*."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    model_device = next(model.parameters()).device
    points = torch.tensor(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1),
        dtype=torch.float32,
        device=model_device,
    )
    control_dim = int(getattr(model, "control_dim", VORTEX_CONTROL_DIM))
    u = torch.zeros((points.shape[0], control_dim), dtype=points.dtype, device=points.device)
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
        mask |= (x_grid - pole_x) ** 2 + y_grid**2 < POLE_MASK_RADIUS_SQ

    ax.streamplot(
        x_grid, y_grid,
        np.ma.array(u_field, mask=mask),
        np.ma.array(v_field, mask=mask),
        density=STREAM_DENSITY,
        linewidth=STREAM_LINE_WIDTH,
        arrowsize=STREAM_ARROW_SIZE,
        color=STREAM_COLOR,
    )
    ax.scatter(
        config.poles, np.zeros(len(config.poles)),
        s=180, facecolors="none", edgecolors="black", linewidths=1.4,
    )


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_stream_and_trajectories(
    states: Tensor,
    config: VortexSystemConfig,
    title: str,
    max_lines: int = 60,
    grid_size: int = 220,
    control_u: float | tuple[float, float] = 0.0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Overlay trajectory bundles on the vortex stream field."""
    states_np = states.detach().cpu().numpy()

    x_vals = states_np[:, :, 0]
    y_vals = states_np[:, :, 1]

    xlim = (
        float(min(np.min(x_vals), min(config.poles)) - 1.0),
        float(max(np.max(x_vals), max(config.poles)) + 1.0),
    )
    ylim = (float(np.min(y_vals) - 1.0), float(np.max(y_vals) + 1.0))

    fig, ax = plt.subplots(figsize=(9, 4))
    _add_stream_field(ax, config, xlim, ylim, grid_size, control_u)

    n = min(max_lines, states_np.shape[0])
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, n))
    for idx in range(n):
        ax.plot(states_np[idx, :, 0], states_np[idx, :, 1], color=shades[idx], lw=1.7, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)


def plot_trajectories(
    states: Tensor,
    title: str,
    max_lines: int = 30,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Simple 2-D trajectory line plot."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for idx in range(min(max_lines, states.shape[0])):
        ax.plot(states[idx, :, 0], states[idx, :, 1], alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)


# ---------------------------------------------------------------------------
# Quiver helpers (shared by comparison and single-trajectory plots)
# ---------------------------------------------------------------------------


def _force_arrows(
    traj_np: np.ndarray,
    controls: Tensor,
    force_gain: tuple[float, float],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quiver origins and (normalised) force vectors."""
    ctrl = controls.detach().cpu().numpy()
    if ctrl.ndim == 1:
        ctrl = ctrl[:, None]
    usable = min(ctrl.shape[0], traj_np.shape[0] - 1)
    if usable <= 0:
        return np.empty((0, 2)), np.empty((0, 2))

    ctrl = ctrl[:usable]
    if ctrl.shape[1] == 1:
        ctrl_x = ctrl[:, 0]
        ctrl_y = ctrl[:, 0]
    else:
        ctrl_x = ctrl[:, 0]
        ctrl_y = ctrl[:, 1]

    vecs = np.column_stack((force_gain[0] * ctrl_x, force_gain[1] * ctrl_y))
    return traj_np[:usable:stride], vecs[::stride]


def _normalise_forces(*vec_arrays: np.ndarray) -> None:
    """Scale force vectors in-place so the longest arrow has length ``FORCE_SCALE_NORM``."""
    max_norm = max(
        (float(np.max(np.linalg.norm(v, axis=1))) if v.size else 0.0)
        for v in vec_arrays
    )
    if max_norm > 1e-12:
        scale = FORCE_SCALE_NORM / max_norm
        for v in vec_arrays:
            v *= scale


# ---------------------------------------------------------------------------
# Comparison plot (real vs. learned optimal trajectories)
# ---------------------------------------------------------------------------


def plot_control_comparison(
    true_opt_traj: Tensor,
    learned_opt_traj: Tensor,
    target: Tensor,
    initial: Tensor,
    controls_true: Tensor | None = None,
    controls_learned: Tensor | None = None,
    force_gain: tuple[float, float] | None = None,
    force_step: int = 3,
    config: VortexSystemConfig | None = None,
    grid_size: int = 220,
    control_u: float | tuple[float, float] = 0.0,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot real-optimal and learned-optimal trajectories side by side."""
    true_np = true_opt_traj.detach().cpu().numpy()
    learned_np = learned_opt_traj.detach().cpu().numpy()
    initial_np = initial.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4.8))

    # Compute bounds
    all_x = np.concatenate([true_np[:, 0], learned_np[:, 0], [initial_np[0], target_np[0]]])
    all_y = np.concatenate([true_np[:, 1], learned_np[:, 1], [initial_np[1], target_np[1]]])
    xlim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)
    ylim = (float(all_y.min()) - 1.0, float(all_y.max()) + 1.0)

    if config is not None:
        xlim = (min(xlim[0], min(config.poles) - 1.0), max(xlim[1], max(config.poles) + 1.0))
        _add_stream_field(ax, config, xlim, ylim, grid_size, control_u)

    ax.plot(true_np[:, 0], true_np[:, 1], lw=2.2, color="#005f73", label="Optimized on real")
    ax.plot(learned_np[:, 0], learned_np[:, 1], lw=2.2, color="#bb3e03", label="Optimized on learned")
    ax.scatter(*initial_np, s=80, marker="o", color="#1f2933", label="Initial")
    ax.scatter(*target_np, s=140, marker="*", color="#ee9b00", label="Target")

    if controls_true is not None and controls_learned is not None and force_gain is not None:
        stride = max(1, force_step)

        pts_t, vecs_t = _force_arrows(true_np, controls_true, force_gain, stride)
        pts_l, vecs_l = _force_arrows(learned_np, controls_learned, force_gain, stride)
        _normalise_forces(vecs_t, vecs_l)

        for pts, vecs, color, label in [
            (pts_t, vecs_t, "#0a9396", "Control force (real-opt)"),
            (pts_l, vecs_l, "#ae2012", "Control force (learned-opt)"),
        ]:
            if pts.shape[0] > 0:
                ax.quiver(
                    pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
                    color=color, alpha=0.95, angles="xy",
                    scale_units="xy", scale=1.0, width=QUIVER_WIDTH, label=label,
                )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)


# ---------------------------------------------------------------------------
# Single optimal-trajectory plot (used by compute_control / optimize scripts)
# ---------------------------------------------------------------------------


def plot_single_control_solution(
    trajectory: Tensor,
    controls: Tensor,
    target: Tensor,
    initial: Tensor,
    force_gain: tuple[float, float],
    config: VortexSystemConfig,
    *,
    title: str = "Optimal control trajectory",
    grid_size: int = 140,
    force_step: int = 3,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot a single optimised trajectory with control-force quiver arrows."""
    traj_np = trajectory.detach().cpu().numpy()
    initial_np = initial.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    poles = np.array(config.poles, dtype=float)

    all_x = np.concatenate([traj_np[:, 0], [initial_np[0], target_np[0]], poles])
    all_y = np.concatenate([traj_np[:, 1], [initial_np[1], target_np[1]]])
    xlim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)
    ylim = (float(all_y.min()) - 1.0, float(all_y.max()) + 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    _add_stream_field(ax, config, xlim, ylim, grid_size)

    ax.plot(traj_np[:, 0], traj_np[:, 1], lw=2.2, color="#005f73", label="Optimal trajectory")
    ax.scatter(*initial_np, s=80, marker="o", color="#1f2933", label="Initial")
    ax.scatter(*target_np, s=140, marker="*", color="#ee9b00", label="Target")

    stride = max(1, force_step)
    pts, vecs = _force_arrows(traj_np, controls, force_gain, stride)
    _normalise_forces(vecs)
    if pts.shape[0] > 0:
        ax.quiver(
            pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
            color="#bb3e03", alpha=0.95, angles="xy",
            scale_units="xy", scale=1.0, width=0.006, label="Control force (scaled)",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)


# ---------------------------------------------------------------------------
# Vector-field error heatmap
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_vector_field_error_map(
    model,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 150,
    t_value: float = 0.0,
    control_u: float | tuple[float, float] = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Return the grid map of L2 drift error ||f_true - f_learned||."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    points = torch.tensor(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1),
        dtype=torch.float32, device=device,
    )
    u_true = _control_tensor_for_dim(
        control_u,
        batch_size=points.shape[0],
        control_dim=VORTEX_CONTROL_DIM,
        dtype=points.dtype,
        device=device,
    )
    model_control_dim = int(getattr(model, "control_dim", VORTEX_CONTROL_DIM))
    u_model = _control_tensor_for_dim(
        control_u,
        batch_size=points.shape[0],
        control_dim=model_control_dim,
        dtype=points.dtype,
        device=device,
    )
    t = torch.tensor(t_value, dtype=points.dtype, device=device)

    true_drift = controlled_vortex_drift(points, u_true, config)
    learned_drift = model.drift_with_control(t=t, y=points, u=u_model)
    error = torch.linalg.vector_norm(learned_drift - true_drift, ord=2, dim=-1)
    return error.reshape(grid_size, grid_size).cpu().numpy()


@torch.no_grad()
def plot_vector_field_error_map(
    model,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 150,
    t_value: float = 0.0,
    control_u: float | tuple[float, float] = 0.0,
    error_vmax: float | None = None,
    device: str = "cpu",
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot an L2-error heatmap comparing learned vs. true drift."""
    error_map = compute_vector_field_error_map(
        model=model,
        config=config,
        xlim=xlim,
        ylim=ylim,
        grid_size=grid_size,
        t_value=t_value,
        control_u=control_u,
        device=device,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    img = ax.imshow(
        error_map, origin="lower",
        extent=(*xlim, *ylim), aspect="equal",
        cmap="magma", vmax=error_vmax,
    )
    ax.scatter(
        config.poles, np.zeros(len(config.poles)),
        s=120, facecolors="none", edgecolors="white", linewidths=1.2,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    u0, u1 = _as_control_pair(control_u)
    ax.set_title(
        f"Vector field error ||f_true - f_learned|| "
        f"(u=({u0:.1f}, {u1:.1f}), t={t_value:.1f})"
    )
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("L2 drift error")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)


def plot_dataset_and_error_grid(
    payloads: list[dict[str, object]],
    models: list,
    labels: list[str],
    *,
    stream_grid: int,
    error_grid: int,
    error_vmax: float | None,
    max_lines: int,
    device: str,
    xlim: tuple[float, float] = (-4.0, 6.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot trajectories (top row) and drift-error maps (bottom row) for each dataset/model pair."""
    if len(payloads) != len(models) or len(models) != len(labels):
        raise ValueError("payloads/models/labels length mismatch.")

    ncols = len(labels)
    configs = [config_from_payload(payload) for payload in payloads]

    fig = plt.figure(figsize=(4.8 * ncols + 1.4, 6.8))
    gs = fig.add_gridspec(
        2,
        ncols + 1,
        width_ratios=[1.0] * ncols + [0.05],
        height_ratios=[1.0, 1.0],
        left=0.05,
        right=0.96,
        bottom=0.08,
        top=0.94,
        wspace=0.08,
        hspace=0.06,
    )
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    cax = fig.add_subplot(gs[:, ncols])
    box_aspect = float((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))

    for idx, (ax, payload, cfg, label) in enumerate(zip(top_axes, payloads, configs, labels, strict=False)):
        states: Tensor = payload["states"]  # type: ignore[assignment]
        _add_stream_field(ax, cfg, xlim, ylim, stream_grid, 0.0)

        arr = states.detach().cpu().numpy()
        n = min(max_lines, arr.shape[0])
        shades = plt.cm.Blues(np.linspace(0.35, 0.95, n))
        for j in range(n):
            ax.plot(arr[j, :, 0], arr[j, :, 1], color=shades[j], lw=1.7, alpha=0.95)

        ax.set_title(f"Dataset {idx + 1} ({label})")
        if idx == 0:
            ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=(idx == 0))

    error_maps = [
        compute_vector_field_error_map(
            model,
            cfg,
            xlim=xlim,
            ylim=ylim,
            grid_size=error_grid,
            device=device,
        )
        for model, cfg in zip(models, configs, strict=False)
    ]

    common_vmin = float(min(float(np.min(m)) for m in error_maps))
    common_vmax = float(max(float(np.max(m)) for m in error_maps)) if error_vmax is None else float(error_vmax)
    if common_vmax <= common_vmin:
        common_vmax = common_vmin + 1e-6

    color_ref = None
    for idx, (ax, err_map, cfg, label) in enumerate(
        zip(bottom_axes, error_maps, configs, labels, strict=False),
    ):
        img = ax.imshow(
            err_map,
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
        if idx == 0:
            ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(axis="both", which="both", labelleft=(idx == 0))

    if color_ref is not None:
        cbar = fig.colorbar(color_ref, cax=cax)
        cbar.set_label("L2 drift error")

    _finalize_plot(fig, save_path, show=show)


def plot_control_comparison_two_rows(
    *,
    config: VortexSystemConfig,
    labels: list[str],
    models: Mapping[str, object],
    x0: Tensor,
    target: Tensor,
    traj_true: Tensor,
    self_trajs: Mapping[str, Tensor],
    transfer_trajs: Mapping[str, Tensor],
    self_controls: Mapping[str, Tensor] | None = None,
    force_gain: tuple[float, float] | None = None,
    force_step: int = 3,
    stream_grid: int = 220,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Top row: per-model simulated optima; bottom row: validation on real dynamics."""
    if len(labels) == 0:
        raise ValueError("labels must contain at least one family.")

    x0_np = x0.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ncols = len(labels)
    fig = plt.figure(figsize=(4.8 * ncols, 8.0))
    gs = fig.add_gridspec(
        2,
        ncols,
        height_ratios=(1.0, 1.0),
        left=0.05,
        right=0.97,
        bottom=0.14,
        top=0.93,
        wspace=0.10,
        hspace=0.10,
    )
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    bottom_ax = fig.add_subplot(gs[1, :])

    all_arrays = [traj_true.detach().cpu().numpy()]
    all_arrays.extend(self_trajs[label].detach().cpu().numpy() for label in labels)
    all_arrays.extend(transfer_trajs[label].detach().cpu().numpy() for label in labels)
    all_x = np.concatenate([arr[:, 0] for arr in all_arrays] + [np.array([x0_np[0], target_np[0], *config.poles])])
    all_y = np.concatenate([arr[:, 1] for arr in all_arrays] + [np.array([x0_np[1], target_np[1]])])
    xlim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)
    ylim = (float(all_y.min()) - 1.0, float(all_y.max()) + 1.0)
    box_aspect = float((ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
    cmap_vals = plt.cm.tab10(np.linspace(0.0, 1.0, max(ncols, 1)))
    family_colors = {label: color for label, color in zip(labels, cmap_vals, strict=False)}

    top_force_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if self_controls is not None and force_gain is not None:
        stride = max(1, force_step)
        vec_arrays: list[np.ndarray] = []
        for label in labels:
            controls = self_controls.get(label)
            if controls is None:
                continue
            traj_np = self_trajs[label].detach().cpu().numpy()
            pts, vecs = _force_arrows(traj_np, controls, force_gain, stride)
            top_force_data[label] = (pts, vecs)
            vec_arrays.append(vecs)
        if vec_arrays:
            _normalise_forces(*vec_arrays)

    for idx, (ax, label) in enumerate(zip(top_axes, labels, strict=False)):
        traj = self_trajs[label].detach().cpu().numpy()
        line_color = family_colors[label]

        _add_model_stream_field(ax, models[label], config, xlim, ylim, stream_grid)
        ax.plot(traj[:, 0], traj[:, 1], lw=2.4, color=line_color, label=f"Optimal on {label}, eval {label}")
        if label in top_force_data:
            pts, vecs = top_force_data[label]
            if pts.shape[0] > 0:
                ax.quiver(
                    pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1],
                    color=line_color, alpha=0.92, angles="xy",
                    scale_units="xy", scale=1.0, width=QUIVER_WIDTH,
                )
        ax.scatter(*x0_np, s=80, marker="o", color="#1f2933")
        ax.scatter(*target_np, s=140, marker="*", color="#ee9b00")
        ax.set_title(f"Simulated ({label})")
        if idx == 0:
            ax.set_ylabel("y")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(box_aspect)
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=(idx == 0))

    _add_stream_field(bottom_ax, config, xlim, ylim, stream_grid, 0.0)
    traj_true_np = traj_true.detach().cpu().numpy()
    bottom_ax.plot(
        traj_true_np[:, 0],
        traj_true_np[:, 1],
        lw=2.8,
        color="#0a9396",
        label="Optimal on real, eval real",
    )

    for label in labels:
        arr = transfer_trajs[label].detach().cpu().numpy()
        bottom_ax.plot(
            arr[:, 0], arr[:, 1], lw=2.2, color=family_colors[label],
            label=f"Optimal on {label}, eval real",
        )

    bottom_ax.scatter(*x0_np, s=80, marker="o", color="#1f2933", label="Initial")
    bottom_ax.scatter(*target_np, s=140, marker="*", color="#ee9b00", label="Target")
    bottom_ax.set_title("Validation on Real Dynamics")
    bottom_ax.set_xlabel("x")
    bottom_ax.set_ylabel("y")
    bottom_ax.set_xlim(*xlim)
    bottom_ax.set_ylim(*ylim)
    bottom_ax.set_aspect("equal", adjustable="box")
    bottom_ax.set_box_aspect(box_aspect)

    handles, legend_labels = bottom_ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)

    _finalize_plot(fig, save_path, show=show)
