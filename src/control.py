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

    horizon_steps: int = 40
    horizon_time: float = 4.0
    solver_dt: float = 0.02
    max_abs_control: float = 4.0
    steps: int = 500
    lr: float = 0.05
    terminal_weight: float = 10.0
    effort_weight: float = 1e-2

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {self.horizon_steps}.")
        if self.horizon_time <= 0.0:
            raise ValueError(f"horizon_time must be > 0, got {self.horizon_time}.")
        if self.solver_dt <= 0.0:
            raise ValueError(f"solver_dt must be > 0, got {self.solver_dt}.")

    @property
    def control_dt(self) -> float:
        """Duration of each piecewise-constant control interval."""
        return self.horizon_time / float(self.horizon_steps)


def rollout(
    sde: ControlledSDE,
    x0: Tensor,
    controls: Tensor,
    solver_dt: float,
    method: str = "euler",
    horizon_time: float | None = None,
) -> Tensor:
    """Integrate an SDE with stepwise-constant controls via ``torchsde``.

    Args:
        sde: Any :class:`ControlledSDE` (real or learned dynamics).
        x0: Initial state ``(state_dim,)``.
        controls: ``(K, control_dim)`` stepwise-constant control sequence.
        solver_dt: Integration time step for the SDE solver.
        horizon_time: Total control horizon in seconds. If ``None``,
            defaults to ``controls.shape[0] * solver_dt`` for backward compatibility.
        method: SDE solver (default ``"euler"``).

    Returns:
        Trajectory ``(K+1, state_dim)`` including the initial state.
    """
    if solver_dt <= 0.0:
        raise ValueError(f"solver_dt must be > 0, got {solver_dt}.")

    K = controls.shape[0]
    if horizon_time is None:
        horizon_time = float(K) * solver_dt
    if horizon_time <= 0.0:
        raise ValueError(f"horizon_time must be > 0, got {horizon_time}.")

    ts = torch.linspace(
        0.0,
        float(horizon_time),
        steps=K + 1,
        dtype=x0.dtype,
        device=x0.device,
    )
    y0 = x0.unsqueeze(0)

    with sde.controlled(ts, controls):
        traj = torchsde.sdeint(sde, y0, ts, method=method, dt=solver_dt)

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
    control_dim = int(getattr(sde, "control_dim", 1))

    raw = torch.zeros(
        (config.horizon_steps, control_dim),
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
        traj = rollout(
            sde,
            x0,
            controls,
            config.solver_dt,
            horizon_time=config.horizon_time,
        )

        term_loss = torch.sum((traj[-1] - target) ** 2)
        effort = torch.mean(controls.square())
        loss = config.terminal_weight * term_loss + config.effort_weight * effort

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

    controls_final = _clamp(raw).detach()
    final_traj = rollout(
        sde,
        x0,
        controls_final,
        config.solver_dt,
        horizon_time=config.horizon_time,
    )
    return controls_final, final_traj.detach(), losses


def terminal_error(trajectory: Tensor, target: Tensor) -> float:
    """L2 distance between the terminal state and the target."""
    return float(torch.linalg.norm(trajectory[-1] - target, ord=2).item())
