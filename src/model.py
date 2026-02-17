from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torchsde
from torch import Tensor
from torch.utils.data import DataLoader


ControlFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class TrainingConfig:
    epochs: int = 250
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    solver_dt: float = 0.02
    method: str = "euler"


class ControlledNeuralSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        state_dim: int = 2,
        control_dim: int = 1,
        hidden_dim: int = 64,
        min_diffusion: float = 1e-3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.min_diffusion = min_diffusion

        in_dim = state_dim + 1
        self.drift_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),
        )

        control_matrix = torch.zeros((state_dim, control_dim), dtype=torch.float32)
        if state_dim >= 2 and control_dim >= 1:
            control_matrix[1, 0] = 1.0
        self.register_buffer("control_matrix", control_matrix)

        self._control_fn: ControlFn = lambda t, x: torch.zeros(
            (x.shape[0], control_dim),
            device=x.device,
            dtype=x.dtype,
        )
        self._control_times: Tensor | None = None
        self._control_values: Tensor | None = None

    def set_control_fn(self, control_fn: ControlFn) -> None:
        self._control_fn = control_fn

    def set_control_trajectory(self, times: Tensor, controls: Tensor) -> None:
        self._control_times = times
        self._control_values = controls

    def clear_control_trajectory(self) -> None:
        self._control_times = None
        self._control_values = None

    def _prepare_features(self, t: Tensor, y: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(y.shape[0])
        t = t.to(device=y.device, dtype=y.dtype).unsqueeze(-1)
        return torch.cat((y, t), dim=-1)

    def _control_from_trajectory(self, t: Tensor, y: Tensor) -> Tensor:
        if self._control_times is None or self._control_values is None:
            return self._control_fn(t, y)

        times = self._control_times.to(device=y.device, dtype=y.dtype)
        controls = self._control_values.to(device=y.device, dtype=y.dtype)

        t_scalar = t.to(device=y.device, dtype=y.dtype)
        if t_scalar.ndim > 0:
            t_scalar = t_scalar.reshape(1)[0]
        t_clamped = torch.clamp(t_scalar, min=times[0], max=times[-1])

        right = int(torch.searchsorted(times, t_clamped, right=False).item())
        if right <= 0:
            return controls[:, 0, :]
        if right >= times.shape[0]:
            return controls[:, -1, :]

        left = right - 1
        t0 = times[left]
        t1 = times[right]
        alpha = (t_clamped - t0) / (t1 - t0 + 1e-12)
        return (1.0 - alpha) * controls[:, left, :] + alpha * controls[:, right, :]

    def drift_with_control(self, t: Tensor, y: Tensor, u: Tensor) -> Tensor:
        features = self._prepare_features(t=t, y=y)
        base_drift = self.drift_net(features)

        if u.ndim == 1:
            u = u.unsqueeze(-1)
        control_term = u @ self.control_matrix.T.to(dtype=y.dtype, device=y.device)
        return base_drift + control_term

    def diffusion_with_control(self, t: Tensor, y: Tensor) -> Tensor:
        features = self._prepare_features(t=t, y=y)
        return self.min_diffusion + self.diffusion_net(features)

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        u = self._control_from_trajectory(t, y)
        return self.drift_with_control(t=t, y=y, u=u)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        return self.diffusion_with_control(t=t, y=y)


def train_neural_sde(
    model: ControlledNeuralSDE,
    dataloader: DataLoader[dict[str, Tensor]],
    times: Tensor,
    config: TrainingConfig,
    val_dataloader: DataLoader[dict[str, Tensor]] | None = None,
    device: str = "cpu",
) -> dict[str, list[float]]:
    model.to(device)
    times = times.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
    }
    for epoch in range(config.epochs):
        model.train()
        running = 0.0

        for batch in dataloader:
            states = batch["states"].to(device)
            controls = batch["controls"].to(device)
            x0 = states[:, 0, :]

            if controls.shape[1] != times.shape[0]:
                if controls.shape[1] == times.shape[0] - 1:
                    controls = torch.cat((controls, controls[:, -1:, :]), dim=1)
                else:
                    raise ValueError(
                        f"Invalid controls shape {controls.shape}; expected T or T-1 with T={times.shape[0]}"
                    )

            model.set_control_trajectory(times=times, controls=controls)

            bm = torchsde.BrownianInterval(
                t0=float(times[0]),
                t1=float(times[-1]),
                size=(states.shape[0], states.shape[-1]),
                dtype=states.dtype,
                device=device,
            )
            preds = torchsde.sdeint(
                sde=model,
                y0=x0,
                ts=times,
                bm=bm,
                method=config.method,
                dt=config.solver_dt,
            )
            preds = preds.transpose(0, 1)

            loss = torch.mean((preds - states) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            model.clear_control_trajectory()
            running += float(loss.item())

        epoch_loss = running / max(len(dataloader), 1)
        history["train_loss"].append(epoch_loss)

        if val_dataloader is not None:
            val_loss = evaluate_neural_sde(
                model=model,
                dataloader=val_dataloader,
                times=times,
                config=config,
                device=device,
            )
            history["val_loss"].append(val_loss)

        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == config.epochs - 1:
            if val_dataloader is None:
                print(f"epoch={epoch + 1:04d} train_loss={epoch_loss:.6f}")
            else:
                print(
                    f"epoch={epoch + 1:04d} train_loss={epoch_loss:.6f} "
                    f"val_loss={history['val_loss'][-1]:.6f}"
                )

    return history


@torch.no_grad()
def evaluate_neural_sde(
    model: ControlledNeuralSDE,
    dataloader: DataLoader[dict[str, Tensor]],
    times: Tensor,
    config: TrainingConfig,
    device: str = "cpu",
) -> float:
    model.eval()
    times = times.to(device)

    running = 0.0
    for batch in dataloader:
        states = batch["states"].to(device)
        controls = batch["controls"].to(device)
        x0 = states[:, 0, :]

        if controls.shape[1] != times.shape[0]:
            if controls.shape[1] == times.shape[0] - 1:
                controls = torch.cat((controls, controls[:, -1:, :]), dim=1)
            else:
                raise ValueError(
                    f"Invalid controls shape {controls.shape}; expected T or T-1 with T={times.shape[0]}"
                )

        model.set_control_trajectory(times=times, controls=controls)

        bm = torchsde.BrownianInterval(
            t0=float(times[0]),
            t1=float(times[-1]),
            size=(states.shape[0], states.shape[-1]),
            dtype=states.dtype,
            device=device,
        )
        preds = torchsde.sdeint(
            sde=model,
            y0=x0,
            ts=times,
            bm=bm,
            method=config.method,
            dt=config.solver_dt,
        )
        preds = preds.transpose(0, 1)

        loss = torch.mean((preds - states) ** 2)
        model.clear_control_trajectory()
        running += float(loss.item())

    return running / max(len(dataloader), 1)


def save_checkpoint(
    model: ControlledNeuralSDE,
    path: str | Path,
    training_config: TrainingConfig,
    history: dict[str, list[float]] | list[float],
) -> None:
    if isinstance(history, list):
        history_payload: dict[str, list[float]] = {
            "train_loss": history,
            "val_loss": [],
        }
    else:
        history_payload = history

    payload = {
        "state_dict": model.state_dict(),
        "model_config": {
            "state_dim": model.state_dim,
            "control_dim": model.control_dim,
            "hidden_dim": int(model.drift_net[0].out_features),
            "min_diffusion": float(model.min_diffusion),
        },
        "training_config": asdict(training_config),
        "loss_history": history_payload,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _infer_model_config(
    model_cfg: object,
    state_dict: Mapping[str, Tensor],
) -> dict[str, int | float]:
    cfg = model_cfg if isinstance(model_cfg, dict) else {}

    control_matrix = state_dict.get("control_matrix")
    state_from_matrix: int | None = None
    control_from_matrix: int | None = None
    if isinstance(control_matrix, torch.Tensor) and control_matrix.ndim == 2:
        state_from_matrix = int(control_matrix.shape[0])
        control_from_matrix = int(control_matrix.shape[1])

    drift_in_weight = state_dict.get("drift_net.0.weight")
    hidden_from_weight: int | None = None
    if isinstance(drift_in_weight, torch.Tensor) and drift_in_weight.ndim == 2:
        hidden_from_weight = int(drift_in_weight.shape[0])

    drift_out_weight = state_dict.get("drift_net.4.weight")
    state_from_output: int | None = None
    if isinstance(drift_out_weight, torch.Tensor) and drift_out_weight.ndim == 2:
        state_from_output = int(drift_out_weight.shape[0])

    state_dim = state_from_matrix or state_from_output or int(cfg.get("state_dim", 2))
    control_dim = control_from_matrix or int(cfg.get("control_dim", 1))
    hidden_dim = hidden_from_weight or int(cfg.get("hidden_dim", 64))
    min_diffusion = float(cfg.get("min_diffusion", 1e-3))

    return {
        "state_dim": state_dim,
        "control_dim": control_dim,
        "hidden_dim": hidden_dim,
        "min_diffusion": min_diffusion,
    }


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[ControlledNeuralSDE, dict[str, object]]:
    loaded = torch.load(Path(path), map_location=device)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        payload: dict[str, object] = loaded
        state_dict = payload["state_dict"]
    elif isinstance(loaded, dict):
        payload = {"state_dict": loaded}
        state_dict = loaded
    else:
        raise TypeError(f"Unsupported checkpoint format in {path}: {type(loaded)}")

    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint state_dict in {path} must be a dict, got {type(state_dict)}")

    model_cfg = _infer_model_config(payload.get("model_config", {}), state_dict)
    model = ControlledNeuralSDE(
        state_dim=int(model_cfg["state_dim"]),
        control_dim=int(model_cfg["control_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        min_diffusion=float(model_cfg["min_diffusion"]),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    return model, payload
