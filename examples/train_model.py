from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import ControlledTrajectoryDataset, config_from_payload, default_control_fn, load_dataset
from src.model import ControlledNeuralSDE, TrainingConfig, save_checkpoint, train_neural_sde
from src.visualization import plot_vector_field_error_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled Neural SDE with torchsde.")
    parser.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    parser.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot-error-map", dest="plot_error_map", action="store_true")
    parser.add_argument("--no-plot-error-map", dest="plot_error_map", action="store_false")
    parser.set_defaults(plot_error_map=True)
    parser.add_argument("--error-map-grid", type=int, default=150)
    parser.add_argument("--image-dir", type=str, default="images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_dataset(args.dataset)
    dataset = ControlledTrajectoryDataset(payload)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must satisfy 0 <= val_ratio < 1, got {args.val_ratio}.")

    val_size = int(len(dataset) * args.val_ratio)
    if args.val_ratio > 0.0 and val_size == 0 and len(dataset) > 1:
        val_size = 1
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_dataloader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        if val_size > 0
        else None
    )

    model = ControlledNeuralSDE(hidden_dim=args.hidden_dim)
    model.set_control_fn(default_control_fn)

    config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        solver_dt=float(payload["times"][1] - payload["times"][0]),
    )
    history = train_neural_sde(
        model=model,
        dataloader=train_dataloader,
        times=payload["times"],
        config=config,
        val_dataloader=val_dataloader,
        device=args.device,
    )
    save_checkpoint(model, args.checkpoint, config, history)

    print(f"train_size={train_size} val_size={val_size} val_ratio={args.val_ratio:.3f}")
    print(f"saved={args.checkpoint}")
    print(f"final_train_loss={history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"final_val_loss={history['val_loss'][-1]:.6f}")

    if args.plot_error_map:
        states = payload["states"]
        if not isinstance(states, torch.Tensor):
            raise TypeError("Payload does not contain valid states tensor.")

        x_values = states[:, :, 0]
        y_values = states[:, :, 1]
        xlim = (float(torch.min(x_values).item()) - 0.5, float(torch.max(x_values).item()) + 0.5)
        ylim = (float(torch.min(y_values).item()) - 0.5, float(torch.max(y_values).item()) + 0.5)

        model.eval()
        plot_vector_field_error_map(
            model=model,
            config=config_from_payload(payload),
            xlim=xlim,
            ylim=ylim,
            grid_size=args.error_map_grid,
            t_value=0.0,
            control_u=0.0,
            device=args.device,
            save_path=Path(args.image_dir) / "train_model_error_map.pdf",
        )


if __name__ == "__main__":
    main()
