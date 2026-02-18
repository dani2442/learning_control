"""Open-loop optimal control via gradient-based optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchsde
from torch import Tensor

from src.sde import ControlledSDE


@dataclass
class OptimalControlConfig:
    """Hyper-parameters for open-loop control optimisation."""

    horizon_steps: int = 80
    dt: float = 0.02
    max_abs_control: float = 4.0
    steps: int = 500
    lr: float = 0.05
    terminal_weight: float = 10.0
    effort_weight: float = 1e-2


def rollout(
    sde: ControlledSDE,
    x0: Tensor,
    controls: Tensor,
    dt: float,
    method: str = "euler",
) -> Tensor:
    """Integrate an SDE with stepwise-constant controls via ``torchsde``.

    Args:
        sde: Any :class:`ControlledSDE` (real or learned dynamics).
        x0: Initial state ``(state_dim,)``.
        controls: ``(K, control_dim)`` stepwise-constant control sequence.
        dt: Integration time step.
        method: SDE solver (default ``"euler"``).

    Returns:
        Trajectory ``(K+1, state_dim)`` including the initial state.
    """
    K = controls.shape[0]
    ts = torch.arange(K + 1, dtype=x0.dtype, device=x0.device) * dt
    y0 = x0.unsqueeze(0)

    with sde.controlled(ts, controls):
        traj = torchsde.sdeint(sde, y0, ts, method=method, dt=dt)

    return traj.squeeze(1)


def optimize_open_loop_controls(
    sde: ControlledSDE,
    x0: Tensor,
    target: Tensor,
    config: OptimalControlConfig,
) -> tuple[Tensor, Tensor, list[float]]:
    """Optimise a stepwise-constant control sequence to steer *x0* toward *target*.

    Returns:
        ``(controls, trajectory, losses)`` — the optimised controls (detached),
        the final trajectory, and the per-step loss history.
    """
    raw = torch.zeros(
        (config.horizon_steps, 1),
        dtype=x0.dtype,
        device=x0.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=config.lr)

    def _clamp(raw_controls: Tensor) -> Tensor:
        return config.max_abs_control * torch.tanh(raw_controls)

    losses: list[float] = []
    for _ in range(config.steps):
        controls = _clamp(raw)
        traj = rollout(sde, x0, controls, config.dt)

        term_loss = torch.sum((traj[-1] - target) ** 2)
        effort = torch.mean(controls.square())
        loss = config.terminal_weight * term_loss + config.effort_weight * effort

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    controls = _clamp(raw)
    final_traj = rollout(sde, x0, controls, config.dt)
    return controls.detach(), final_traj.detach(), losses


def terminal_error(trajectory: Tensor, target: Tensor) -> float:
    """L2 distance between the terminal state and the target."""
    return float(torch.linalg.norm(trajectory[-1] - target, ord=2).item())
