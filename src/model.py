"""Learned neural SDE model: architecture, training, evaluation, and checkpointing."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchsde
from torch import Tensor
from torch.utils.data import DataLoader

from src.controls import ControlFn
from src.sde import ControlledSDE

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 250
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    solver_dt: float = 0.02
    method: str = "euler"


class ControlledNeuralSDE(ControlledSDE):
    """Learned SDE whose drift is parameterised by a neural network."""

    def __init__(
        self,
        state_dim: int = 2,
        control_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__(control_dim=control_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        in_dim = state_dim + 1  # state + time
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.drift_net = nn.Sequential(*layers)

        control_matrix = torch.zeros((state_dim, control_dim), dtype=torch.float32)
        if state_dim >= 2 and control_dim == 1:
            # Legacy scalar-control fallback.
            control_matrix[1, 0] = 1.0
        else:
            for idx in range(min(state_dim, control_dim)):
                control_matrix[idx, idx] = 1.0
        self.register_buffer("control_matrix", control_matrix)

        self._control_fn: ControlFn = lambda t, x: torch.zeros(
            (x.shape[0], control_dim), device=x.device, dtype=x.dtype,
        )

    def set_control_fn(self, control_fn: ControlFn) -> None:
        self._control_fn = control_fn

    def _lookup_control(self, t: Tensor, y: Tensor) -> Tensor:
        """Zero-order hold with fallback to the callable control function."""
        if self._control_times is None or self._control_values is None:
            u = self._control_fn(t, y)
            return u.reshape(y.shape[0], self.control_dim)
        return super()._lookup_control(t, y)

    def _prepare_features(self, t: Tensor, y: Tensor) -> Tensor:
        time = t.to(device=y.device, dtype=y.dtype).reshape(-1, 1)
        if time.shape[0] == 1:
            time = time.expand(y.shape[0], 1)
        return torch.cat((y, time), dim=-1)

    def drift_with_control(self, t: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Compute the drift: neural_net(y, t) + B @ sin(u)."""
        base_drift = self.drift_net(self._prepare_features(t, y))
        u = u.reshape(-1, self.control_dim)
        if u.shape[0] == 1 and y.shape[0] > 1:
            u = u.expand(y.shape[0], self.control_dim)
        control_term = torch.sin(u) @ self.control_matrix.T.to(dtype=y.dtype, device=y.device)
        return base_drift + control_term

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        u = self._lookup_control(t, y)
        return self.drift_with_control(t=t, y=y, u=u)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_neural_sde(
    model: ControlledNeuralSDE,
    dataloader: DataLoader[dict[str, Tensor]],
    times: Tensor,
    config: TrainingConfig,
    val_dataloader: DataLoader[dict[str, Tensor]] | None = None,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Train the neural SDE on trajectory data."""
    model.to(device)
    times = times.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        running = 0.0

        for batch in dataloader:
            states = batch["states"].to(device)
            controls = batch["controls"].to(device)
            x0 = states[:, 0, :]

            with model.controlled(times, controls):
                traj = torchsde.sdeint(model, x0, times, method=config.method, dt=config.solver_dt)
            preds = traj.permute(1, 0, 2)
            loss = torch.mean((preds - states) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            running += float(loss.item())

        epoch_loss = running / max(len(dataloader), 1)
        history["train_loss"].append(epoch_loss)

        if val_dataloader is not None:
            val_loss = evaluate_neural_sde(model, val_dataloader, times, config, device)
            history["val_loss"].append(val_loss)

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == config.epochs - 1:
            msg = f"epoch={epoch + 1:04d} train_loss={epoch_loss:.6f}"
            if val_dataloader is not None:
                msg += f" val_loss={history['val_loss'][-1]:.6f}"
            logger.info(msg)

    return history


@torch.no_grad()
def evaluate_neural_sde(
    model: ControlledNeuralSDE,
    dataloader: DataLoader[dict[str, Tensor]],
    times: Tensor,
    config: TrainingConfig,
    device: str = "cpu",
) -> float:
    """Evaluate mean squared prediction error on a dataloader."""
    model.eval()
    times = times.to(device)

    running = 0.0
    for batch in dataloader:
        states = batch["states"].to(device)
        controls = batch["controls"].to(device)
        x0 = states[:, 0, :]

        with model.controlled(times, controls):
            traj = torchsde.sdeint(model, x0, times, method=config.method, dt=config.solver_dt)
        preds = traj.permute(1, 0, 2)
        running += float(torch.mean((preds - states) ** 2).item())

    return running / max(len(dataloader), 1)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: ControlledNeuralSDE,
    path: str | Path,
    training_config: TrainingConfig,
    history: dict[str, list[float]],
) -> None:
    """Save model weights, architecture config, and loss history."""
    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "state_dim": model.state_dim,
            "control_dim": model.control_dim,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
        },
        "training_config": asdict(training_config),
        "loss_history": history,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[ControlledNeuralSDE, dict[str, object]]:
    """Load a model from a checkpoint file."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = payload["model_config"]
    model = ControlledNeuralSDE(
        state_dim=int(cfg["state_dim"]),
        control_dim=int(cfg["control_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
    )
    # strict=False: ignore legacy keys (e.g. diffusion_net) from older checkpoints
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device)
    return model, payload
