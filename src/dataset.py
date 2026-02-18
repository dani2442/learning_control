"""Dataset generation, persistence, and loading for controlled trajectory data."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torchsde
from torch import Tensor
from torch.utils.data import Dataset

from src.controls import ControlFn, _as_control_tensor
from src.dynamics import VortexSDE, VortexSystemConfig

# Legacy-compatible type alias (control factories now live in src.controls)
SimpleControlFn = Callable[[Tensor], Tensor | float]


# ---------------------------------------------------------------------------
# Path tagging helpers
# ---------------------------------------------------------------------------


def control_type_from_payload(payload: dict[str, object], *, default: str = "unknown") -> str:
    """Extract the control-type string stored inside a dataset payload."""
    control = payload.get("control")
    if not isinstance(control, dict):
        return default
    control_type = control.get("type")
    if isinstance(control_type, str) and control_type.strip():
        return control_type.strip()
    return default


def _sanitise_tag(control_type: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in control_type.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def add_control_type_to_path(
    path: str | Path,
    control_type: str,
    *,
    as_subdir: bool = False,
) -> Path:
    """Append a sanitised control-type tag to *path* (filename suffix or subdirectory)."""
    p = Path(path)
    tag = _sanitise_tag(control_type)

    if as_subdir:
        return p if p.parent.name == tag else p.parent / tag / p.name

    stem = p.stem if p.suffix else p.name
    if stem == tag or stem.endswith(f"_{tag}"):
        return p
    name = f"{stem}_{tag}{p.suffix}" if p.suffix else f"{stem}_{tag}"
    return p.with_name(name)


# ---------------------------------------------------------------------------
# SDE integration
# ---------------------------------------------------------------------------


def rollout_controlled_system(
    initial_states: Tensor,
    times: Tensor,
    control_fn: ControlFn | SimpleControlFn,
    config: VortexSystemConfig,
) -> tuple[Tensor, Tensor]:
    """Integrate the controlled vortex system via ``torchsde.sdeint``.

    Controls are pre-computed (open-loop) at each interval boundary, then
    applied as a stepwise-constant schedule during SDE integration.
    """
    batch_size = initial_states.shape[0]

    # Pre-compute open-loop controls at each interval start
    control_list = []
    for step in range(times.numel() - 1):
        u_t = _as_control_tensor(
            control_fn(times[step], initial_states),
            batch_size=batch_size,
            reference=initial_states,
        )
        control_list.append(u_t)
    controls = torch.stack(control_list, dim=1)  # (batch, K, control_dim)

    sde = VortexSDE(config)
    dt = float(times[1] - times[0])
    with sde.controlled(times, controls):
        traj = torchsde.sdeint(sde, initial_states, times, method="euler", dt=dt)

    # sdeint returns (T, batch, state_dim) -> (batch, T, state_dim)
    states = traj.permute(1, 0, 2)
    return states, controls


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    num_trajectories: int = 512,
    horizon: float = 4.0,
    dt: float = 0.02,
    state_box: tuple[float, float, float, float] = (-4.0, 4.0, -2.0, 2.0),
    seed: int = 7,
    config: VortexSystemConfig | None = None,
    control_fn: ControlFn | SimpleControlFn | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Generate a trajectory dataset for controlled dynamics."""
    cfg = config or VortexSystemConfig()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if control_fn is None:
        # Sensible default: zero control
        control_fn = lambda t, x: torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

    xmin, xmax, ymin, ymax = state_box
    x0 = torch.empty((num_trajectories, 2), device=device)
    x0[:, 0].uniform_(xmin, xmax, generator=gen)
    x0[:, 1].uniform_(ymin, ymax, generator=gen)

    steps = int(round(horizon / dt)) + 1
    times = torch.linspace(0.0, horizon, steps=steps, device=device)

    states, controls = rollout_controlled_system(
        initial_states=x0,
        times=times,
        control_fn=control_fn,
        config=cfg,
    )

    return {
        "times": times.cpu(),
        "states": states.cpu(),
        "controls": controls.cpu(),
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_dataset(payload: dict[str, object], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_dataset(path: str | Path) -> dict[str, object]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ControlledTrajectoryDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset wrapping a trajectory payload."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.times: Tensor = payload["times"]  # type: ignore[assignment]
        self.states: Tensor = payload["states"]  # type: ignore[assignment]
        self.controls: Tensor = payload["controls"]  # type: ignore[assignment]

        if not all(isinstance(t, Tensor) for t in (self.times, self.states, self.controls)):
            raise TypeError("Dataset payload has invalid tensor fields.")

    @classmethod
    def from_file(cls, path: str | Path) -> ControlledTrajectoryDataset:
        return cls(load_dataset(path))

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"states": self.states[index], "controls": self.controls[index]}


# ---------------------------------------------------------------------------
# Config reconstruction
# ---------------------------------------------------------------------------


def config_from_payload(payload: dict[str, object]) -> VortexSystemConfig:
    """Reconstruct a :class:`VortexSystemConfig` from a saved payload."""
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
