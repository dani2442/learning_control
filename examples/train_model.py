"""Train a controlled Neural SDE on trajectory data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    ControlledTrajectoryDataset,
    add_control_type_to_path,
    config_from_payload,
    control_type_from_payload,
    load_dataset,
)
from src.model import ControlledNeuralSDE, TrainingConfig, save_checkpoint, train_neural_sde
from src.visualization import plot_vector_field_error_map

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled neural dynamics model (deterministic drift only).")
    parser.add_argument("--dataset", type=str, default="data/controlled_vortex_constant.pt")
    parser.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot-error-map", dest="plot_error_map", action="store_true")
    parser.add_argument("--no-plot-error-map", dest="plot_error_map", action="store_false")
    parser.set_defaults(plot_error_map=True)
    parser.add_argument("--show-plot", dest="show_plot", action="store_true")
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false")
    parser.set_defaults(show_plot=False)
    parser.add_argument("--error-map-grid", type=int, default=150)
    parser.add_argument("--image-dir", type=str, default="images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_dataset(args.dataset)
    control_type = control_type_from_payload(payload)
    dataset = ControlledTrajectoryDataset(payload)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must satisfy 0 <= val_ratio < 1, got {args.val_ratio}.")

    val_size = max(1, round(len(dataset) * args.val_ratio)) if args.val_ratio > 0. else 0
    val_size = min(val_size, len(dataset) - 1)
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
    checkpoint_path = add_control_type_to_path(args.checkpoint, control_type)
    save_checkpoint(model, checkpoint_path, config, history)

    print(f"train_size={train_size} val_size={val_size} val_ratio={args.val_ratio:.3f}")
    print(f"saved={checkpoint_path}")
    print(f"final_train_loss={history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"final_val_loss={history['val_loss'][-1]:.6f}")

    if args.plot_error_map:
        model.eval()
        error_map_path = add_control_type_to_path(
            Path(args.image_dir) / "train_model_error_map.pdf",
            control_type,
            as_subdir=True,
        )
        plot_vector_field_error_map(
            model=model,
            config=config_from_payload(payload),
            xlim=(-4,4),
            ylim=(-2,2),
            grid_size=args.error_map_grid,
            t_value=0.0,
            control_u=0.0,
            error_vmax=5.0,
            device=args.device,
            save_path=error_map_path,
            show=args.show_plot,
        )


if __name__ == "__main__":
    main()
