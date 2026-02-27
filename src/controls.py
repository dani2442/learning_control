"""Control function factories and related type definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

# Canonical control-function type: (time, state) -> control tensor.
ControlFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class RandomSinusoidalControlConfig:
    """Parameters for random per-trajectory sinusoidal controls."""

    amplitude: float = 0.6
    frequency_range: tuple[float, float] = (0.01, 0.2)
    phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    bias: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_generator(
    *,
    seed: int | None,
    generator: torch.Generator | None,
    device: str,
) -> torch.Generator:
    if generator is not None:
        return generator
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def _as_control_tensor(value: Tensor | float, *, batch_size: int, reference: Tensor) -> Tensor:
    """Normalise a control value into shape ``(batch, control_dim)``."""
    return _as_control_tensor_with_dim(
        value,
        batch_size=batch_size,
        control_dim=2,
        reference=reference,
    )


def _as_control_tensor_with_dim(
    value: Tensor | float,
    *,
    batch_size: int,
    control_dim: int,
    reference: Tensor,
) -> Tensor:
    """Normalise a control value into shape ``(batch, control_dim)``."""
    if not isinstance(value, Tensor):
        value = torch.tensor(value, dtype=reference.dtype, device=reference.device)
    t = value.to(dtype=reference.dtype, device=reference.device)

    if t.ndim == 0:
        return t.reshape(1, 1).expand(batch_size, control_dim)

    if t.ndim == 1:
        if t.shape[0] == control_dim:
            return t.unsqueeze(0).expand(batch_size, control_dim)
        return t.reshape(batch_size, control_dim)

    if t.ndim == 2:
        if t.shape == (1, control_dim):
            return t.expand(batch_size, control_dim)
        if t.shape == (batch_size, 1):
            return t.expand(batch_size, control_dim)
        if t.shape == (1, 1):
            return t.expand(batch_size, control_dim)
        return t.reshape(batch_size, control_dim)

    return t.reshape(batch_size, control_dim)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_random_sinusoidal_control_fn(
    *,
    num_trajectories: int,
    control_dim: int = 2,
    config: RandomSinusoidalControlConfig | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """Build ``u(t) = A * sin(2*pi*f*t + phase) + bias`` with random f/phase per trajectory."""
    cfg = config or RandomSinusoidalControlConfig()
    gen = _make_generator(seed=seed, generator=generator, device=device)

    frequencies = torch.empty((num_trajectories, control_dim), device=device)
    phases = torch.empty((num_trajectories, control_dim), device=device)
    frequencies.uniform_(cfg.frequency_range[0], cfg.frequency_range[1], generator=gen)
    phases.uniform_(cfg.phase_range[0], cfg.phase_range[1], generator=gen)

    def control_fn(t: Tensor, _x: Tensor | None = None) -> Tensor:
        time = t.to(dtype=frequencies.dtype, device=frequencies.device).reshape(-1, 1)
        if time.shape[0] == 1:
            time = time.expand(num_trajectories, 1)
        u = cfg.amplitude * torch.sin(
            2.0 * math.pi * frequencies * time + phases
        ) + cfg.bias
        return u

    meta: dict[str, object] = {
        "type": "sinusoidal",
        "amplitude": cfg.amplitude,
        "frequency_range": cfg.frequency_range,
        "phase_range": cfg.phase_range,
        "bias": cfg.bias,
        "control_dim": control_dim,
        "seed": seed,
    }
    return control_fn, meta


def make_constant_control_fn(
    *,
    num_trajectories: int,
    value: float | list[float] | tuple[float, ...] | Tensor = 0.0,
    control_dim: int = 2,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """Build a constant (time-invariant) control, optionally sampled per trajectory."""
    gen = _make_generator(seed=seed, generator=generator, device=device)

    if isinstance(value, Tensor):
        raw = value.to(device=device, dtype=torch.float32)
    elif isinstance(value, (list, tuple)):
        raw = torch.tensor(value, device=device, dtype=torch.float32)
    else:
        raw = torch.tensor(float(value), device=device, dtype=torch.float32)

    if raw.ndim == 0:
        candidates = raw.reshape(1, 1).expand(1, control_dim)
    elif raw.ndim == 1:
        if raw.numel() == control_dim:
            candidates = raw.reshape(1, control_dim)
        else:
            candidates = raw.reshape(-1, 1).expand(-1, control_dim)
    else:
        candidates = raw.reshape(-1, raw.shape[-1])
        if candidates.shape[1] == 1:
            candidates = candidates.expand(-1, control_dim)

    if candidates.shape[0] == 1:
        constants = candidates.expand(num_trajectories, control_dim)
    else:
        indices = torch.randint(
            0, candidates.shape[0], (num_trajectories,),
            generator=gen, device=candidates.device,
        )
        constants = candidates[indices]

    def control_fn(t: Tensor, _x: Tensor | None = None) -> Tensor:
        return constants.to(dtype=t.dtype, device=t.device)

    values_list = [[float(v) for v in row] for row in candidates.detach().cpu().tolist()]
    meta: dict[str, object] = {"type": "constant", "values": values_list, "control_dim": control_dim}
    if len(values_list) == 1:
        meta["value"] = values_list[0]
    else:
        meta["assignment"] = "uniform_random_per_trajectory"
        meta["seed"] = seed
    return control_fn, meta
