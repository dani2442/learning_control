from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.dataset import VortexSystemConfig, controlled_vortex_drift
from src.model import ControlledNeuralSDE


@dataclass
class OptimalControlConfig:
    horizon_steps: int = 80
    dt: float = 0.02
    max_abs_control: float = 2.5
    steps: int = 500
    lr: float = 0.05
    terminal_weight: float = 10.0
    effort_weight: float = 1e-2


def rollout_real_dynamics(
    x0: Tensor,
    controls: Tensor,
    dt: float,
    system_config: VortexSystemConfig,
) -> Tensor:
    x = x0
    traj = [x]

    for k in range(controls.shape[0]):
        u_k = controls[k].view(1, 1)
        drift = controlled_vortex_drift(x.view(1, -1), u_k, system_config).view(-1)
        x = x + dt * drift
        traj.append(x)

    return torch.stack(traj, dim=0)


def rollout_learned_dynamics(
    model: ControlledNeuralSDE,
    x0: Tensor,
    controls: Tensor,
    dt: float,
) -> Tensor:
    x = x0
    traj = [x]

    for k in range(controls.shape[0]):
        t = torch.tensor(k * dt, dtype=x.dtype, device=x.device)
        u_k = controls[k].view(1, 1)
        drift = model.drift_with_control(t=t, y=x.view(1, -1), u=u_k).view(-1)
        x = x + dt * drift
        traj.append(x)

    return torch.stack(traj, dim=0)


def optimize_open_loop_controls(
    rollout_fn,
    x0: Tensor,
    target: Tensor,
    config: OptimalControlConfig,
    device: str,
) -> tuple[Tensor, Tensor, list[float]]:
    raw_controls = torch.zeros(
        (config.horizon_steps, 1),
        dtype=x0.dtype,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw_controls], lr=config.lr)

    losses: list[float] = []
    for _ in range(config.steps):
        controls = config.max_abs_control * torch.tanh(raw_controls)
        trajectory = rollout_fn(x0, controls, config.dt)

        terminal_error = torch.sum((trajectory[-1] - target) ** 2)
        effort = torch.mean(controls.square())
        loss = config.terminal_weight * terminal_error + config.effort_weight * effort

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    controls = config.max_abs_control * torch.tanh(raw_controls)
    final_traj = rollout_fn(x0, controls, config.dt)
    return controls.detach(), final_traj.detach(), losses


def terminal_error(trajectory: Tensor, target: Tensor) -> float:
    return float(torch.linalg.norm(trajectory[-1] - target, ord=2).item())
