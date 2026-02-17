from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from src.dataset import VortexSystemConfig, controlled_vortex_drift


def _vortex_stream_field(
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int,
    control_u: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    u_field = np.full_like(x_grid, config.background_speed, dtype=float)
    v_field = np.zeros_like(y_grid, dtype=float)
    mask = np.zeros_like(x_grid, dtype=bool)

    for pole_x, gamma in zip(config.poles, config.strengths, strict=True):
        dx = x_grid - pole_x
        r2 = dx * dx + y_grid * y_grid + config.eps

        u_field += gamma * (2.0 * y_grid) / r2
        v_field += -gamma * (2.0 * dx) / r2

        mask |= r2 < 0.15**2

    u_field += config.control_gain_x * control_u
    v_field += config.control_gain_y * control_u

    speed = np.hypot(u_field, v_field)
    cap = 6.0
    scale = np.minimum(1.0, cap / (speed + 1e-12))
    u_field = u_field * scale
    v_field = v_field * scale

    return x_grid, y_grid, u_field, v_field, mask


def _finalize_plot(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf")
    plt.show()


def plot_stream_and_trajectories(
    states: Tensor,
    config: VortexSystemConfig,
    title: str,
    max_lines: int = 60,
    grid_size: int = 220,
    control_u: float = 0.0,
    save_path: str | Path | None = None,
) -> None:
    states_np = states.detach().cpu().numpy()

    x_values = states_np[:, :, 0]
    y_values = states_np[:, :, 1]

    x_min = float(min(np.min(x_values), min(config.poles)) - 1.0)
    x_max = float(max(np.max(x_values), max(config.poles)) + 1.0)
    y_min = float(np.min(y_values) - 1.0)
    y_max = float(np.max(y_values) + 1.0)

    x_grid, y_grid, u_field, v_field, mask = _vortex_stream_field(
        config=config,
        xlim=(x_min, x_max),
        ylim=(y_min, y_max),
        grid_size=grid_size,
        control_u=control_u,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.streamplot(
        x_grid,
        y_grid,
        np.ma.array(u_field, mask=mask),
        np.ma.array(v_field, mask=mask),
        density=1.45,
        linewidth=0.6,
        arrowsize=0.9,
        color="#9aa6b2",
    )

    n_lines = min(max_lines, states_np.shape[0])
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, n_lines))
    for idx in range(n_lines):
        ax.plot(states_np[idx, :, 0], states_np[idx, :, 1], color=shades[idx], lw=1.7, alpha=0.95)

    ax.scatter(config.poles, np.zeros(len(config.poles)), s=180, facecolors="none", edgecolors="black", linewidths=1.4)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _finalize_plot(fig, save_path)


def plot_trajectories(
    states: Tensor,
    title: str,
    max_lines: int = 30,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for idx in range(min(max_lines, states.shape[0])):
        ax.plot(states[idx, :, 0], states[idx, :, 1], alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    _finalize_plot(fig, save_path)


def plot_control_comparison(
    true_opt_traj: Tensor,
    learned_opt_traj: Tensor,
    target: Tensor,
    initial: Tensor,
    config: VortexSystemConfig | None = None,
    grid_size: int = 220,
    control_u: float = 0.0,
    save_path: str | Path | None = None,
) -> None:
    true_np = true_opt_traj.detach().cpu().numpy()
    learned_np = learned_opt_traj.detach().cpu().numpy()
    initial_np = initial.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x_min = min(true_np[:, 0].min(), learned_np[:, 0].min(), float(initial_np[0]), float(target_np[0])) - 1.0
    x_max = max(true_np[:, 0].max(), learned_np[:, 0].max(), float(initial_np[0]), float(target_np[0])) + 1.0
    y_min = min(true_np[:, 1].min(), learned_np[:, 1].min(), float(initial_np[1]), float(target_np[1])) - 1.0
    y_max = max(true_np[:, 1].max(), learned_np[:, 1].max(), float(initial_np[1]), float(target_np[1])) + 1.0

    if config is not None:
        x_min = min(x_min, min(config.poles) - 1.0)
        x_max = max(x_max, max(config.poles) + 1.0)
        x_grid, y_grid, u_field, v_field, mask = _vortex_stream_field(
            config=config,
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
            grid_size=grid_size,
            control_u=control_u,
        )
        ax.streamplot(
            x_grid,
            y_grid,
            np.ma.array(u_field, mask=mask),
            np.ma.array(v_field, mask=mask),
            density=1.45,
            linewidth=0.6,
            arrowsize=0.9,
            color="#9aa6b2",
        )
        ax.scatter(config.poles, np.zeros(len(config.poles)), s=180, facecolors="none", edgecolors="black", linewidths=1.4)

    ax.plot(true_np[:, 0], true_np[:, 1], label="Optimized on real", lw=2)
    ax.plot(learned_np[:, 0], learned_np[:, 1], label="Optimized on learned", lw=2)
    ax.scatter(initial_np[0], initial_np[1], marker="o", s=70, label="Initial")
    ax.scatter(target_np[0], target_np[1], marker="*", s=120, label="Target")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    _finalize_plot(fig, save_path)


@torch.no_grad()
def plot_vector_field_error_map(
    model,
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 150,
    t_value: float = 0.0,
    control_u: float = 0.0,
    device: str = "cpu",
    save_path: str | Path | None = None,
) -> None:
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    points = np.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    points_t = torch.tensor(points, dtype=torch.float32, device=device)
    controls = torch.full((points_t.shape[0], 1), fill_value=control_u, dtype=points_t.dtype, device=device)
    t = torch.tensor(t_value, dtype=points_t.dtype, device=device)

    true_drift = controlled_vortex_drift(points_t, controls, config)
    learned_drift = model.drift_with_control(t=t, y=points_t, u=controls)
    error = torch.linalg.vector_norm(learned_drift - true_drift, ord=2, dim=-1)
    error_map = error.reshape(grid_size, grid_size).cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    image = ax.imshow(
        error_map,
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="equal",
        cmap="magma",
    )
    ax.scatter(config.poles, np.zeros(len(config.poles)), s=120, facecolors="none", edgecolors="white", linewidths=1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Vector field error ||f_true - f_learned|| (u={control_u:.1f}, t={t_value:.1f})")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("L2 drift error")
    plt.tight_layout()
    _finalize_plot(fig, save_path)
