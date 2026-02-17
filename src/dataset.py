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
    diffusion: float = 0.0


@dataclass(frozen=True)
class RandomSinusoidalControlConfig:
    amplitude: float = 0.6
    frequency_range: tuple[float, float] = (0.4, 2.0)
    phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    bias: float = 0.0


ControlFn = Callable[[Tensor], Tensor | float]


def default_control_fn(t: Tensor) -> Tensor:
    """Default time-only control used to generate the training dataset."""
    return 0.6 * torch.sin(0.7 * t)


def control_type_from_payload(payload: dict[str, object], *, default: str = "unknown") -> str:
    control = payload.get("control")
    if not isinstance(control, dict):
        return default

    control_type = control.get("type")
    if isinstance(control_type, str) and control_type.strip():
        return control_type.strip()
    return default


def control_type_to_filename_tag(control_type: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in control_type.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def add_control_type_to_path(path: str | Path, control_type: str) -> Path:
    path_obj = Path(path)
    tag = control_type_to_filename_tag(control_type)
    stem = path_obj.stem if path_obj.suffix else path_obj.name
    if stem == tag or stem.endswith(f"_{tag}"):
        return path_obj

    filename = f"{stem}_{tag}{path_obj.suffix}" if path_obj.suffix else f"{stem}_{tag}"
    return path_obj.with_name(filename)


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
    local_generator = torch.Generator(device=device)
    if seed is not None:
        local_generator.manual_seed(seed)
    return local_generator


def _as_control_tensor(
    control_value: Tensor | float,
    *,
    batch_size: int,
    reference: Tensor,
) -> Tensor:
    if isinstance(control_value, Tensor):
        u = control_value.to(dtype=reference.dtype, device=reference.device)
    else:
        u = torch.tensor(control_value, dtype=reference.dtype, device=reference.device)

    if u.ndim == 0:
        return u.expand(batch_size, 1)

    if u.ndim == 1:
        if u.shape[0] == 1:
            return u.expand(batch_size).unsqueeze(-1)
        if u.shape[0] == batch_size:
            return u.unsqueeze(-1)

    if u.ndim == 2 and u.shape[1] == 1:
        if u.shape[0] == 1:
            return u.expand(batch_size, 1)
        if u.shape[0] == batch_size:
            return u

    raise ValueError(
        "Control output must be scalar, shape (1,), (batch,), (1, 1), or (batch, 1); "
        f"got {tuple(u.shape)} for batch={batch_size}."
    )


def make_random_sinusoidal_control_fn(
    *,
    num_trajectories: int,
    config: RandomSinusoidalControlConfig | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    """Build u_control(t) = A * sin(2*pi*f*t + phase) + b with random f and phase."""
    cfg = config or RandomSinusoidalControlConfig()
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")

    _validate_range("frequency_range", cfg.frequency_range)
    _validate_range("phase_range", cfg.phase_range)
    local_generator = _make_generator(seed=seed, generator=generator, device=device)

    sample_device = torch.device(device)
    frequencies = torch.empty((num_trajectories,), device=sample_device)
    phases = torch.empty((num_trajectories,), device=sample_device)
    frequencies.uniform_(cfg.frequency_range[0], cfg.frequency_range[1], generator=local_generator)
    phases.uniform_(cfg.phase_range[0], cfg.phase_range[1], generator=local_generator)

    def control_fn(t: Tensor) -> Tensor:
        time = t.to(dtype=frequencies.dtype, device=frequencies.device)
        if time.ndim == 0:
            time = time.expand(num_trajectories)
        elif time.ndim == 1 and time.shape[0] == 1:
            time = time.expand(num_trajectories)
        elif time.ndim != 1 or time.shape[0] != num_trajectories:
            raise ValueError(
                f"Expected t to be scalar or shape ({num_trajectories},), got {tuple(time.shape)}."
            )
        u = cfg.amplitude * torch.sin(2.0 * math.pi * frequencies * time + phases) + cfg.bias
        return u.unsqueeze(-1)

    control_meta: dict[str, object] = {
        "type": "sinusoidal",
        "amplitude": cfg.amplitude,
        "frequency_range": cfg.frequency_range,
        "phase_range": cfg.phase_range,
        "bias": cfg.bias,
        "seed": seed,
    }
    return control_fn, control_meta


def make_constant_control_fn(
    *,
    num_trajectories: int,
    value: float | list[float] | tuple[float, ...] | Tensor = 0.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: str = "cpu",
) -> tuple[ControlFn, dict[str, object]]:
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive, got {num_trajectories}.")
    local_generator = _make_generator(seed=seed, generator=generator, device=device)

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
            low=0,
            high=candidates.numel(),
            size=(num_trajectories,),
            generator=local_generator,
            device=torch.device(device),
        )
        constants = candidates[indices].unsqueeze(-1)

    def control_fn(t: Tensor) -> Tensor:
        time = t if isinstance(t, Tensor) else torch.tensor(t, device=constants.device)
        return constants.to(dtype=time.dtype, device=time.device)

    values_list = [float(v) for v in candidates.detach().cpu().tolist()]
    control_meta: dict[str, object] = {"type": "constant", "values": values_list}
    if len(values_list) == 1:
        control_meta["value"] = values_list[0]
    else:
        control_meta["assignment"] = "uniform_random_per_trajectory"
        control_meta["seed"] = seed
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
    stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Euler rollout for many trajectories in parallel."""
    states = [initial_states]
    controls: list[Tensor] = []

    x_t = initial_states
    dt = times[1] - times[0]
    sqrt_dt = torch.sqrt(dt)

    for step in range(times.numel() - 1):
        t = times[step]
        u_t = _as_control_tensor(control_fn(t), batch_size=x_t.shape[0], reference=x_t)
        drift = controlled_vortex_drift(x_t, u_t, config)

        if stochastic and config.diffusion != 0.0:
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
        stochastic=False,
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
