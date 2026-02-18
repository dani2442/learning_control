"""Plotting utilities for trajectories, stream fields, and error maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from src.dynamics import (
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
    control_u: float = 0.0,
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


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def plot_stream_and_trajectories(
    states: Tensor,
    config: VortexSystemConfig,
    title: str,
    max_lines: int = 60,
    grid_size: int = 220,
    control_u: float = 0.0,
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
    ctrl = controls.detach().cpu().numpy().ravel()
    usable = min(ctrl.shape[0], traj_np.shape[0] - 1)
    if usable <= 0:
        return np.empty((0, 2)), np.empty((0, 2))

    ctrl = ctrl[:usable]
    vecs = np.column_stack((force_gain[0] * ctrl, force_gain[1] * ctrl))
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
    control_u: float = 0.0,
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
def plot_vector_field_error_map(
    model,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 150,
    t_value: float = 0.0,
    control_u: float = 0.0,
    error_vmax: float | None = None,
    device: str = "cpu",
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot an L2-error heatmap comparing learned vs. true drift."""
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    points = torch.tensor(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1),
        dtype=torch.float32, device=device,
    )
    u = torch.full((points.shape[0], 1), control_u, dtype=points.dtype, device=device)
    t = torch.tensor(t_value, dtype=points.dtype, device=device)

    true_drift = controlled_vortex_drift(points, u, config)
    learned_drift = model.drift_with_control(t=t, y=points, u=u)

    error = torch.linalg.vector_norm(learned_drift - true_drift, ord=2, dim=-1)
    error_map = error.reshape(grid_size, grid_size).cpu().numpy()

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
    ax.set_title(f"Vector field error ||f_true - f_learned|| (u={control_u:.1f}, t={t_value:.1f})")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("L2 drift error")
    plt.tight_layout()
    _finalize_plot(fig, save_path, show=show)
