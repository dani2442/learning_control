"""Compare control-input families for learning quality and control transfer."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control import OptimalControlConfig, optimize_open_loop_controls, rollout, terminal_error
from src.dataset import (
    ControlledTrajectoryDataset,
    add_control_type_to_path,
    generate_dataset,
    save_dataset,
)
from src.dynamics import VORTEX_CONTROL_DIM, VortexSystemConfig, VortexSDE
from src.model import (
    ControlledNeuralSDE,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_neural_sde,
)
from src.visualization import (
    compute_vector_field_error_map,
    plot_control_comparison_two_rows,
    plot_dataset_and_error_grid,
)


INPUT_FAMILIES = ("constant", "sinusoidal")  # , "stepwise")

EVAL_XLIM = (-4.0, 6.0)
EVAL_YLIM = (-3.0, 3.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate/train/compare control-input families.")

    p.add_argument("--dataset", type=str, default="data/controlled_vortex.pt")
    p.add_argument("--checkpoint", type=str, default="data/neural_sde.pt")

    p.add_argument("--num-trajectories", type=int, default=512)
    p.add_argument("--dataset-horizon", type=float, default=2.0, help="Dataset trajectory duration in seconds.")
    p.add_argument("--dataset-dt", type=float, default=0.05, help="Dataset time spacing used for generated trajectories.")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--control-rms", type=float, default=0.6, help="Target RMS level over the dataset horizon. All trajectories are normalized to this RMS.")
    p.add_argument("--sin-freq-range", type=float, nargs=2, default=(0.4, 2.0))
    p.add_argument("--sin-phase-range", type=float, nargs=2, default=(0.0, 2.0 * math.pi))
    p.add_argument("--step-segments", type=int, default=8)

    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--mlp-layers", type=int, default=1, help="Number of hidden layers in the neural drift MLP.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument(
        "--use-checkpoint",
        dest="use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip training and load models from --checkpoint (tagged by control family).",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--pole-exclusion-radius",
        type=float,
        default=0.25,
        help="Reject initial states within this radius of any pole (set 0 to disable).",
    )

    p.add_argument("--x0", type=float, nargs=2, default=(-3.0, 1.2))
    p.add_argument("--target", type=float, nargs=2, default=(3.5, 0.0))
    p.add_argument("--control-horizon-steps", type=int, default=OptimalControlConfig.horizon_steps, help="Number of piecewise-constant control intervals for optimization.")
    p.add_argument(
        "--control-horizon-time",
        type=float,
        default=OptimalControlConfig.horizon_time,
        help="Total control horizon in seconds for optimization.",
    )
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--max-abs-control", type=float, default=6.0)
    p.add_argument("--control-lr", type=float, default=OptimalControlConfig.lr)
    p.add_argument("--terminal-weight", type=float, default=30.0)
    p.add_argument("--effort-weight", type=float, default=1e-3)

    p.add_argument("--max-lines", type=int, default=60)
    p.add_argument("--stream-grid", type=int, default=220)
    p.add_argument("--error-grid", type=int, default=150)
    p.add_argument("--error-vmax", type=float, default=2)
    p.add_argument("--force-step", type=int, default=3)
    p.add_argument("--image-dir", type=str, default="images")
    p.add_argument("--show-plot", dest="show_plot", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


def _time_grid(horizon: float, dt: float) -> Tensor:
    steps = int(round(horizon / dt))
    return torch.linspace(0.0, horizon, steps=steps + 1, dtype=torch.float32)


def _normalize_controls_to_l2(
    controls_flat: Tensor,
    *,
    dt: float,
    target_l2: float,
) -> Tensor:
    l2 = torch.sqrt(dt * torch.sum(controls_flat.square(), dim=1, keepdim=True)).clamp_min(1e-8)
    return controls_flat * (target_l2 / l2)


def _control_l2_per_trajectory(controls: Tensor, dt: float) -> Tensor:
    flat = controls.reshape(controls.shape[0], -1)
    return torch.sqrt(dt * torch.sum(flat.square(), dim=1))


def _make_scheduled_control_fn(times: Tensor, controls: Tensor):
    """Create a control callable that serves a precomputed stepwise schedule."""
    times_cpu = times.detach().cpu()
    controls_cpu = controls.detach().cpu()

    def control_fn(t: Tensor, _x: Tensor | None = None) -> Tensor:
        t_cpu = t.detach().to(device=times_cpu.device, dtype=times_cpu.dtype).squeeze()
        idx = int(torch.clamp(
            torch.searchsorted(times_cpu, t_cpu, right=True) - 1,
            0,
            controls_cpu.shape[1] - 1,
        ).item())
        return controls_cpu[:, idx, :].to(dtype=t.dtype, device=t.device)

    return control_fn


def _build_control_schedule(
    args: argparse.Namespace,
    family: str,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    times = _time_grid(args.dataset_horizon, args.dataset_dt)
    step_times = times[:-1]
    steps = step_times.numel()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if family == "constant":
        signs = torch.randint(
            0, 2, (args.num_trajectories, VORTEX_CONTROL_DIM), generator=gen
        )
        raw = (2 * signs - 1).to(torch.float32).unsqueeze(1).expand(
            args.num_trajectories,
            steps,
            VORTEX_CONTROL_DIM,
        )
        family_meta: dict[str, object] = {
            "type": "constant",
            "sign_assignment": "uniform_random_{-1,+1}",
            "control_dim": VORTEX_CONTROL_DIM,
            "seed": seed,
        }
    elif family == "sinusoidal":
        freq = torch.empty((args.num_trajectories, VORTEX_CONTROL_DIM))
        phase = torch.empty((args.num_trajectories, VORTEX_CONTROL_DIM))
        freq.uniform_(float(args.sin_freq_range[0]), float(args.sin_freq_range[1]), generator=gen)
        phase.uniform_(float(args.sin_phase_range[0]), float(args.sin_phase_range[1]), generator=gen)
        t = step_times.unsqueeze(0).unsqueeze(-1)
        raw = torch.sin(2.0 * math.pi * freq.unsqueeze(1) * t + phase.unsqueeze(1))
        family_meta = {
            "type": "sinusoidal",
            "frequency_range": (float(args.sin_freq_range[0]), float(args.sin_freq_range[1])),
            "phase_range": (float(args.sin_phase_range[0]), float(args.sin_phase_range[1])),
            "control_dim": VORTEX_CONTROL_DIM,
            "seed": seed,
        }
    elif family == "stepwise":
        segments = max(1, min(int(args.step_segments), steps))
        segment_values = torch.randn(
            (args.num_trajectories, segments, VORTEX_CONTROL_DIM),
            generator=gen,
        )
        step_to_segment = torch.div(torch.arange(steps) * segments, steps, rounding_mode="floor")
        raw = segment_values[:, step_to_segment]
        family_meta = {
            "type": "stepwise",
            "segments": segments,
            "control_dim": VORTEX_CONTROL_DIM,
            "seed": seed,
        }
    else:
        raise ValueError(f"Unsupported family: {family}")

    target_l2 = float(args.control_rms * math.sqrt(args.dataset_horizon))
    controls_flat = _normalize_controls_to_l2(
        raw.reshape(args.num_trajectories, -1),
        dt=args.dataset_dt,
        target_l2=target_l2,
    )
    controls = controls_flat.reshape(args.num_trajectories, steps, VORTEX_CONTROL_DIM)
    l2_vals = _control_l2_per_trajectory(controls, args.dataset_dt)

    meta = {
        **family_meta,
        "target_rms": float(args.control_rms),
        "target_l2": target_l2,
        "l2_mean": float(l2_vals.mean().item()),
        "l2_std": float(l2_vals.std(unbiased=False).item()),
    }
    return times, controls, meta


def _generate_one_dataset(
    args: argparse.Namespace,
    *,
    control_type: str,
    seed: int,
    config: VortexSystemConfig,
) -> tuple[dict[str, object], Path]:
    times, controls, control_meta = _build_control_schedule(args, control_type, seed)
    control_fn = _make_scheduled_control_fn(times, controls)

    payload = generate_dataset(
        num_trajectories=args.num_trajectories,
        horizon=args.dataset_horizon,
        dt=args.dataset_dt,
        pole_exclusion_radius=args.pole_exclusion_radius,
        seed=seed,
        config=config,
        control_fn=control_fn,
    )
    payload["control"] = control_meta

    out_path = add_control_type_to_path(args.dataset, control_type)
    save_dataset(payload, out_path)
    return payload, out_path


def _split_loaders(
    payload: dict[str, object],
    *,
    batch_size: int,
    val_ratio: float,
    split_seed: int,
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]] | None]:
    dataset = ControlledTrajectoryDataset(payload)
    val_size = max(1, round(len(dataset) * val_ratio)) if val_ratio > 0.0 else 0
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_size > 0 else None
    return train_loader, val_loader


def _train_one_model(
    args: argparse.Namespace,
    payload: dict[str, object],
    control_type: str,
) -> tuple[ControlledNeuralSDE, dict[str, list[float]], Path]:
    train_loader, val_loader = _split_loaders(
        payload,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    controls = payload["controls"]  # type: ignore[assignment]
    control_dim = int(controls.shape[-1])
    model = ControlledNeuralSDE(
        control_dim=control_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.mlp_layers,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        solver_dt=float(payload["times"][1] - payload["times"][0]),
    )
    history = train_neural_sde(
        model=model,
        dataloader=train_loader,
        times=payload["times"],
        config=train_cfg,
        val_dataloader=val_loader,
        device=args.device,
    )
    model.eval()

    checkpoint_path = add_control_type_to_path(args.checkpoint, control_type)
    save_checkpoint(model, checkpoint_path, train_cfg, history)
    return model, history, checkpoint_path


def _print_metric_table(rows: list[dict[str, float | str]]) -> None:
    print("Family comparison metrics:")
    print(
        f"{'family':<12} {'drift_mean':>11} {'drift_p95':>11} "
        f"{'opt->self':>11} {'opt->real':>11}"
    )
    for row in rows:
        print(
            f"{row['family']:<12} "
            f"{row['drift_mean']:>11.4f} "
            f"{row['drift_p95']:>11.4f} "
            f"{row['self_error']:>11.4f} "
            f"{row['transfer_error']:>11.4f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    config = VortexSystemConfig()

    payloads: dict[str, dict[str, object]] = {}
    models: dict[str, ControlledNeuralSDE] = {}

    for i, family in enumerate(INPUT_FAMILIES):
        payload, dataset_path = _generate_one_dataset(
            args,
            control_type=family,
            seed=args.seed + i,
            config=config,
        )
        payloads[family] = payload

        l2_vals = _control_l2_per_trajectory(payload["controls"], args.dataset_dt)  # type: ignore[index]
        print(
            f"dataset_{family}={dataset_path} "
            f"control_l2_mean={float(l2_vals.mean().item()):.4f} "
            f"control_l2_std={float(l2_vals.std(unbiased=False).item()):.4e}"
        )

        if args.use_checkpoint:
            checkpoint_path = add_control_type_to_path(args.checkpoint, family)
            model, ckpt_payload = load_checkpoint(checkpoint_path, device=args.device)
            model.eval()
            models[family] = model
            train_hist = ckpt_payload.get("loss_history", {}).get("train_loss", [])
            val_hist = ckpt_payload.get("loss_history", {}).get("val_loss", [])
            final_train = float(train_hist[-1]) if train_hist else float("nan")
            final_val = float(val_hist[-1]) if val_hist else float("nan")
            print(
                f"checkpoint_{family}={checkpoint_path} "
                f"loaded_train_loss={final_train:.6f} loaded_val_loss={final_val:.6f}"
            )
        else:
            model, history, checkpoint_path = _train_one_model(args, payload, family)
            models[family] = model
            final_train = history["train_loss"][-1]
            final_val = history["val_loss"][-1] if history["val_loss"] else float("nan")
            print(
                f"checkpoint_{family}={checkpoint_path} "
                f"final_train_loss={final_train:.6f} final_val_loss={final_val:.6f}"
            )

    labels = list(INPUT_FAMILIES)
    payload_list = [payloads[label] for label in labels]
    model_list = [models[label] for label in labels]

    grid_path = Path(args.image_dir) / "three_datasets_three_error_maps.pdf"
    plot_dataset_and_error_grid(
        payloads=payload_list,
        models=model_list,
        labels=labels,
        stream_grid=args.stream_grid,
        error_grid=args.error_grid,
        error_vmax=args.error_vmax,
        max_lines=args.max_lines,
        device=args.device,
        xlim=EVAL_XLIM,
        ylim=EVAL_YLIM,
        save_path=grid_path,
        show=args.show_plot,
    )
    print(f"saved={grid_path}")

    dts = [float(payload["times"][1] - payload["times"][0]) for payload in payload_list]

    control_cfg = OptimalControlConfig(
        horizon_steps=args.control_horizon_steps,
        horizon_time=args.control_horizon_time,
        solver_dt=dts[0],
        max_abs_control=args.max_abs_control,
        steps=args.iters,
        lr=args.control_lr,
        terminal_weight=args.terminal_weight,
        effort_weight=args.effort_weight,
    )
    print(
        "control_cfg "
        f"horizon_steps={control_cfg.horizon_steps} "
        f"horizon_time={control_cfg.horizon_time:.4f} "
        f"control_dt={control_cfg.control_dt:.4f} "
        f"solver_dt={control_cfg.solver_dt:.4f} "
        f"iters={control_cfg.steps} "
        f"max_abs_control={control_cfg.max_abs_control:.2f} "
        f"terminal_weight={control_cfg.terminal_weight:.3f} "
        f"effort_weight={control_cfg.effort_weight:.6f}"
    )

    x0 = torch.tensor(args.x0, dtype=torch.float32, device=args.device)
    target = torch.tensor(args.target, dtype=torch.float32, device=args.device)
    real_sde = VortexSDE(config)

    _, traj_true, _ = optimize_open_loop_controls(real_sde, x0, target, control_cfg)

    self_controls: dict[str, Tensor] = {}
    self_trajs: dict[str, Tensor] = {}
    transfer_trajs: dict[str, Tensor] = {}
    rows: list[dict[str, float | str]] = []
    for family in labels:
        model = models[family]
        controls, traj_self, _ = optimize_open_loop_controls(model, x0, target, control_cfg)
        self_controls[family] = controls
        self_trajs[family] = traj_self
        traj_real = rollout(
            real_sde,
            x0,
            controls,
            control_cfg.solver_dt,
            horizon_time=control_cfg.horizon_time,
        )
        transfer_trajs[family] = traj_real

        err_map = compute_vector_field_error_map(
            model,
            config,
            xlim=EVAL_XLIM,
            ylim=EVAL_YLIM,
            grid_size=args.error_grid,
            device=args.device,
        )
        rows.append(
            {
                "family": family,
                "drift_mean": float(np.mean(err_map)),
                "drift_p95": float(np.quantile(err_map, 0.95)),
                "self_error": terminal_error(traj_self, target),
                "transfer_error": terminal_error(traj_real, target),
            }
        )

    print("Terminal L2 errors:")
    print(f"  optimize real, eval real: {terminal_error(traj_true, target):.4f}")
    for row in rows:
        print(f"  optimize {row['family']}, eval {row['family']}: {row['self_error']:.4f}")
        print(f"  optimize {row['family']}, eval real: {row['transfer_error']:.4f}")

    _print_metric_table(rows)

    control_path = Path(args.image_dir) / "three_way_control_comparison.pdf"
    plot_control_comparison_two_rows(
        config=config,
        labels=labels,
        models=models,
        x0=x0,
        target=target,
        traj_true=traj_true,
        self_controls=self_controls,
        self_trajs=self_trajs,
        transfer_trajs=transfer_trajs,
        force_gain=(float(config.control_gain_x), float(config.control_gain_y)),
        force_step=args.force_step,
        stream_grid=args.stream_grid,
        save_path=control_path,
        show=args.show_plot,
    )
    print(f"saved={control_path}")


if __name__ == "__main__":
    main()
