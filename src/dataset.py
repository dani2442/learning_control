from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class VortexSystemConfig:
    background_speed: float = 1.0
    poles: tuple[float, float, float] = (-2.0, 0.0, 2.0)
    strengths: tuple[float, float, float] = (1.0, -1.0, 1.0)
    eps: float = 1e-3
    control_gain_x: float = 0.05
    control_gain_y: float = 0.5
    diffusion: float = 0.08


@dataclass(frozen=True)
class RandomControlBasisConfig:
    basis: str = "sinusoidal"
    num_terms: int = 3
    amplitude_range: tuple[float, float] = (0.2, 0.8)
    frequency_range: tuple[float, float] = (0.4, 2.0)
    phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    coefficient_scale: float = 0.6
    bias_range: tuple[float, float] = (-0.2, 0.2)


ControlFn = Callable[[Tensor, Tensor], Tensor]


def default_control_fn(t: Tensor, x: Tensor) -> Tensor:
    """Baseline control policy used to generate the training dataset."""
    t = t.to(dtype=x.dtype, device=x.device)
    if t.ndim == 0:
        t = t.expand(x.shape[0])
    u = 0.6 * torch.sin(0.7 * t) + 0.15 * x[:, 0] - 0.25 * x[:, 1]
    return u.unsqueeze(-1)


def _validate_range(name: str, bounds: tuple[float, float]) -> None:
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name} must satisfy low <= high, got {bounds}.")


def make_random_basis_control_fn(
    *,
    num_trajectories: int,
    horizon: float,
    config: RandomControlBasisConfig | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """
    Build a random per-trajectory control policy from a selected basis.

    Supported bases:
    - sinusoidal: sum_j a_j * sin(2*pi*f_j*t + phi_j) + b
    - polynomial: sum_j c_j * (t / horizon)^j + b
    """
    cfg = config or RandomControlBasisConfig()
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")
    if cfg.num_terms <= 0:
        raise ValueError(f"num_terms must be positive, got {cfg.num_terms}.")
    if horizon <= 0.0:
        raise ValueError(f"horizon must be positive, got {horizon}.")

    supported_bases = {"sinusoidal", "polynomial"}
    if cfg.basis not in supported_bases:
        raise ValueError(f"Unsupported basis '{cfg.basis}'. Supported bases: {sorted(supported_bases)}.")

    _validate_range("amplitude_range", cfg.amplitude_range)
    _validate_range("frequency_range", cfg.frequency_range)
    _validate_range("phase_range", cfg.phase_range)
    _validate_range("bias_range", cfg.bias_range)

    local_generator = generator
    if local_generator is None:
        local_generator = torch.Generator(device=device)
        if seed is not None:
            local_generator.manual_seed(seed)
    elif seed is not None:
        raise ValueError("Provide either seed or generator, not both.")

    sample_device = torch.device(device)
    basis_kind = cfg.basis

    bias = torch.empty((num_trajectories,), device=sample_device)
    bias.uniform_(cfg.bias_range[0], cfg.bias_range[1], generator=local_generator)

    if basis_kind == "sinusoidal":
        amplitudes = torch.empty((num_trajectories, cfg.num_terms), device=sample_device)
        frequencies = torch.empty((num_trajectories, cfg.num_terms), device=sample_device)
        phases = torch.empty((num_trajectories, cfg.num_terms), device=sample_device)

        amplitudes.uniform_(cfg.amplitude_range[0], cfg.amplitude_range[1], generator=local_generator)
        frequencies.uniform_(cfg.frequency_range[0], cfg.frequency_range[1], generator=local_generator)
        phases.uniform_(cfg.phase_range[0], cfg.phase_range[1], generator=local_generator)

        def control_fn(t: Tensor, x: Tensor) -> Tensor:
            t = t.to(dtype=x.dtype, device=x.device)
            if t.ndim == 0:
                t = t.expand(x.shape[0])
            if t.shape[0] != x.shape[0]:
                raise ValueError(f"Expected t to have shape ({x.shape[0]},) or scalar, got {tuple(t.shape)}.")

            amp = amplitudes.to(device=x.device, dtype=x.dtype)
            freq = frequencies.to(device=x.device, dtype=x.dtype)
            phase = phases.to(device=x.device, dtype=x.dtype)
            bias_local = bias.to(device=x.device, dtype=x.dtype)

            angle = 2.0 * math.pi * t.unsqueeze(-1) * freq + phase
            u = torch.sum(amp * torch.sin(angle), dim=-1) + bias_local
            return u.unsqueeze(-1)

        control_meta: dict[str, object] = {
            "type": "random_basis",
            "basis": basis_kind,
            "num_terms": cfg.num_terms,
            "amplitude_range": cfg.amplitude_range,
            "frequency_range": cfg.frequency_range,
            "phase_range": cfg.phase_range,
            "bias_range": cfg.bias_range,
            "seed": seed,
        }
        return control_fn, control_meta

    coefficients = torch.empty((num_trajectories, cfg.num_terms), device=sample_device)
    coefficients.uniform_(-cfg.coefficient_scale, cfg.coefficient_scale, generator=local_generator)
    powers = torch.arange(cfg.num_terms, device=sample_device, dtype=torch.float32)

    def control_fn(t: Tensor, x: Tensor) -> Tensor:
        t = t.to(dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t to have shape ({x.shape[0]},) or scalar, got {tuple(t.shape)}.")

        coeff = coefficients.to(device=x.device, dtype=x.dtype)
        powers_local = powers.to(device=x.device, dtype=x.dtype)
        bias_local = bias.to(device=x.device, dtype=x.dtype)

        tau = torch.clamp(t / float(horizon), min=0.0, max=1.0)
        features = tau.unsqueeze(-1).pow(powers_local)
        u = torch.sum(coeff * features, dim=-1) + bias_local
        return u.unsqueeze(-1)

    control_meta = {
        "type": "random_basis",
        "basis": basis_kind,
        "num_terms": cfg.num_terms,
        "coefficient_scale": cfg.coefficient_scale,
        "bias_range": cfg.bias_range,
        "seed": seed,
    }
    return control_fn, control_meta


def controlled_vortex_drift(
    x: Tensor,
    u: Tensor,
    config: VortexSystemConfig,
) -> Tensor:
    """Compute x' = f(x, u) for the controlled vortex system."""
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


def rollout_controlled_system(
    initial_states: Tensor,
    times: Tensor,
    control_fn: ControlFn,
    config: VortexSystemConfig,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Euler-Maruyama rollout for many trajectories in parallel."""
    states = [initial_states]
    controls: list[Tensor] = []

    x_t = initial_states
    dt = times[1] - times[0]
    sqrt_dt = torch.sqrt(dt)

    for step in range(times.numel() - 1):
        t = times[step]
        u_t = control_fn(t, x_t)
        drift = controlled_vortex_drift(x_t, u_t, config)

        if stochastic:
            noise = torch.randn(
                x_t.shape,
                dtype=x_t.dtype,
                device=x_t.device,
                generator=generator,
            )
            noise = config.diffusion * sqrt_dt * noise
        else:
            noise = torch.zeros_like(x_t)

        x_t = x_t + dt * drift + noise
        states.append(x_t)
        controls.append(u_t)

    return torch.stack(states, dim=1), torch.stack(controls, dim=1)


def generate_dataset(
    num_trajectories: int = 512,
    horizon: float = 4.0,
    dt: float = 0.02,
    state_box: tuple[float, float, float, float] = (-4.0, 4.0, -2.0, 2.0),
    seed: int = 7,
    config: VortexSystemConfig | None = None,
    control_fn: ControlFn = default_control_fn,
    device: str = "cpu",
) -> dict[str, object]:
    """Generate trajectory dataset for controlled dynamics."""
    cfg = config or VortexSystemConfig()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    xmin, xmax, ymin, ymax = state_box
    x0 = torch.empty((num_trajectories, 2), device=device)
    x0[:, 0].uniform_(xmin, xmax, generator=generator)
    x0[:, 1].uniform_(ymin, ymax, generator=generator)

    steps = int(round(horizon / dt)) + 1
    times = torch.linspace(0.0, horizon, steps=steps, device=device)

    states, controls = rollout_controlled_system(
        initial_states=x0,
        times=times,
        control_fn=control_fn,
        config=cfg,
        stochastic=True,
        generator=generator,
    )

    return {
        "times": times.cpu(),
        "states": states.cpu(),
        "controls": controls.cpu(),
        "config": asdict(cfg),
    }


def save_dataset(payload: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_dataset(path: str | Path) -> dict[str, object]:
    return torch.load(Path(path), map_location="cpu")


class ControlledTrajectoryDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, payload: dict[str, object]) -> None:
        self.times = payload["times"]
        self.states = payload["states"]
        self.controls = payload["controls"]

        if not isinstance(self.times, Tensor) or not isinstance(self.states, Tensor) or not isinstance(self.controls, Tensor):
            raise TypeError("Dataset payload has invalid tensor fields.")

    @classmethod
    def from_file(cls, path: str | Path) -> "ControlledTrajectoryDataset":
        payload = load_dataset(path)
        return cls(payload)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "states": self.states[index],
            "controls": self.controls[index],
        }


def config_from_payload(payload: dict[str, object]) -> VortexSystemConfig:
    raw = payload.get("config")
    if not isinstance(raw, dict):
        return VortexSystemConfig()
    return VortexSystemConfig(
        background_speed=float(raw["background_speed"]),
        poles=tuple(raw["poles"]),
        strengths=tuple(raw["strengths"]),
        eps=float(raw["eps"]),
        control_gain_x=float(raw["control_gain_x"]),
        control_gain_y=float(raw["control_gain_y"]),
        diffusion=float(raw["diffusion"]),
    )
