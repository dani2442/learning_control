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


def _validate_range(name: str, bounds: tuple[float, float]) -> None:
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name} must satisfy low <= high, got {bounds}.")


def _make_generator(
    *,
    seed: int | None,
    generator: torch.Generator | None,
    device: str,
) -> torch.Generator:
    if generator is not None and seed is not None:
        raise ValueError("Provide either seed or generator, not both.")
    if generator is not None:
        return generator
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def _as_control_tensor(value: Tensor | float, *, batch_size: int, reference: Tensor) -> Tensor:
    """Normalise an arbitrary control value into shape ``(batch, 1)``."""
    if not isinstance(value, Tensor):
        value = torch.tensor(value, dtype=reference.dtype, device=reference.device)
    t = value.to(dtype=reference.dtype, device=reference.device)
    return t.reshape(-1, 1).expand(batch_size, 1)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_random_sinusoidal_control_fn(
    *,
    num_trajectories: int,
    config: RandomSinusoidalControlConfig | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """Build ``u(t) = A * sin(2*pi*f*t + phase) + bias`` with random f/phase per trajectory."""
    cfg = config or RandomSinusoidalControlConfig()
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")

    _validate_range("frequency_range", cfg.frequency_range)
    _validate_range("phase_range", cfg.phase_range)
    gen = _make_generator(seed=seed, generator=generator, device=device)

    frequencies = torch.empty(num_trajectories, device=device)
    phases = torch.empty(num_trajectories, device=device)
    frequencies.uniform_(cfg.frequency_range[0], cfg.frequency_range[1], generator=gen)
    phases.uniform_(cfg.phase_range[0], cfg.phase_range[1], generator=gen)

    def control_fn(t: Tensor, _x: Tensor | None = None) -> Tensor:
        time = t.to(dtype=frequencies.dtype, device=frequencies.device)
        if time.numel() == 1:
            time = time.expand(num_trajectories)
        u = cfg.amplitude * torch.sin(2.0 * math.pi * frequencies * time + phases) + cfg.bias
        return u.unsqueeze(-1)

    meta: dict[str, object] = {
        "type": "sinusoidal",
        "amplitude": cfg.amplitude,
        "frequency_range": cfg.frequency_range,
        "phase_range": cfg.phase_range,
        "bias": cfg.bias,
        "seed": seed,
    }
    return control_fn, meta


def make_constant_control_fn(
    *,
    num_trajectories: int,
    value: float | list[float] | tuple[float, ...] | Tensor = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """Build a constant (time-invariant) control, optionally sampled per trajectory."""
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")
    gen = _make_generator(seed=seed, generator=generator, device=device)

    if isinstance(value, Tensor):
        candidates = value.flatten().to(device=device, dtype=torch.float32)
    elif isinstance(value, (list, tuple)):
        candidates = torch.tensor(value, device=device, dtype=torch.float32).flatten()
    else:
        candidates = torch.tensor([float(value)], device=device, dtype=torch.float32)

    if candidates.numel() == 0:
        raise ValueError("value must include at least one constant.")

    if candidates.numel() == 1:
        constants = candidates.expand(num_trajectories).unsqueeze(-1)
    else:
        indices = torch.randint(
            0, candidates.numel(), (num_trajectories,),
            generator=gen, device=torch.device(device),
        )
        constants = candidates[indices].unsqueeze(-1)

    def control_fn(t: Tensor, _x: Tensor | None = None) -> Tensor:
        return constants.to(dtype=t.dtype, device=t.device)

    values_list = [float(v) for v in candidates.detach().cpu().tolist()]
    meta: dict[str, object] = {"type": "constant", "values": values_list}
    if len(values_list) == 1:
        meta["value"] = values_list[0]
    else:
        meta["assignment"] = "uniform_random_per_trajectory"
        meta["seed"] = seed
    return control_fn, meta
