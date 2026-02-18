"""Vortex system dynamics: configuration, drift, SDE wrapper, and vector field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from src.sde import ControlledSDE


@dataclass(frozen=True)
class VortexSystemConfig:
    """Physical parameters for the controlled vortex flow."""

    background_speed: float = 1.0
    poles: tuple[float, float, float] = (-2.0, 0.0, 2.0)
    strengths: tuple[float, float, float] = (1.0, -1.0, 1.0)
    eps: float = 1e-3
    control_gain_x: float = 0.05
    control_gain_y: float = 0.5
    diffusion: float = 0.0


def controlled_vortex_drift(
    x: Tensor,
    u: Tensor,
    config: VortexSystemConfig,
) -> Tensor:
    """Compute drift f(x, u) for the controlled vortex system.

    Args:
        x: State tensor ``(batch, 2)``.
        u: Control tensor ``(batch, 1)``.
        config: System parameters.

    Returns:
        Drift ``(batch, 2)``.
    """
    poles = torch.tensor(config.poles, dtype=x.dtype, device=x.device)
    strengths = torch.tensor(config.strengths, dtype=x.dtype, device=x.device)

    dx = x[:, 0:1] - poles
    y = x[:, 1:2]
    r2 = dx.square() + y.square() + config.eps

    base_x = config.background_speed + torch.sum(strengths * (2.0 * y) / r2, dim=-1, keepdim=True)
    base_y = torch.sum(-strengths * (2.0 * dx) / r2, dim=-1, keepdim=True)

    drift_x = base_x + config.control_gain_x * u
    drift_y = base_y + config.control_gain_y * u
    return torch.cat((drift_x, drift_y), dim=-1)


class VortexSDE(ControlledSDE):
    """``torchsde``-compatible wrapper for the real controlled vortex system."""

    def __init__(self, config: VortexSystemConfig) -> None:
        super().__init__(control_dim=1)
        self.config = config

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        u = self._lookup_control(t, y)
        return controlled_vortex_drift(y, u, self.config)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        return torch.full_like(y, self.config.diffusion)


# ---------------------------------------------------------------------------
# Grid-based vector field (single implementation used by all plotting code)
# ---------------------------------------------------------------------------

POLE_MASK_RADIUS_SQ = 0.15**2
SPEED_CAP = 6.0


def vortex_vector_field(
    config: VortexSystemConfig,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_size: int = 220,
    control_u: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the vortex drift on a regular grid (for stream / quiver plots).

    Returns:
        ``(x_grid, y_grid, u_field, v_field, pole_mask)`` — all ``(grid_size, grid_size)``.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    x_grid, y_grid = np.meshgrid(xs, ys)

    points = torch.tensor(
        np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1),
        dtype=torch.float32,
    )
    controls = torch.full((points.shape[0], 1), control_u)

    drift = controlled_vortex_drift(points, controls, config).numpy()
    u_field = drift[:, 0].reshape(grid_size, grid_size)
    v_field = drift[:, 1].reshape(grid_size, grid_size)

    # Cap extreme speeds for cleaner visualisation
    speed = np.hypot(u_field, v_field)
    scale = np.minimum(1.0, SPEED_CAP / (speed + 1e-12))
    u_field *= scale
    v_field *= scale

    # Mask near poles
    mask = np.zeros_like(x_grid, dtype=bool)
    for pole_x in config.poles:
        mask |= (x_grid - pole_x) ** 2 + y_grid**2 < POLE_MASK_RADIUS_SQ

    return x_grid, y_grid, u_field, v_field, mask
