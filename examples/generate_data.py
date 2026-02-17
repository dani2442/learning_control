from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Callable

from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import (
    RandomSinusoidalControlConfig,
    config_from_payload,
    generate_dataset,
    make_constant_control_fn,
    make_random_sinusoidal_control_fn,
    save_dataset,
)
from src.visualization import plot_stream_and_trajectories


def load_control_fn(spec: str) -> Callable[[Tensor], Tensor | float]:
    if ":" not in spec:
        raise ValueError("Control function must be provided as 'module_path:function_name'.")
    module_name, function_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable.")
    return fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled trajectory dataset.")
    parser.add_argument("--output", type=str, default="data/controlled_vortex.pt")
    parser.add_argument("--num-trajectories", type=int, default=512)
    parser.add_argument("--horizon", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--control-type",
        choices=("sinusoidal", "constant", "custom"),
        default="sinusoidal",
        help="Control example to use.",
    )
    parser.add_argument(
        "--control-fn",
        type=str,
        default=None,
        help="Custom control callable as 'module_path:function_name' with signature u_control(t).",
    )
    parser.add_argument("--sin-amplitude", type=float, default=0.6)
    parser.add_argument("--sin-freq-range", type=float, nargs=2, default=(0.4, 2.0))
    parser.add_argument("--sin-phase-range", type=float, nargs=2, default=(0.0, 2.0 * math.pi))
    parser.add_argument("--sin-bias", type=float, default=0.0)
    parser.add_argument("--constant-value", type=float, default=0.0)
    parser.add_argument("--plot", default=True, action="store_true")
    parser.add_argument("--max-lines", type=int, default=60)
    parser.add_argument("--image-dir", type=str, default="images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_meta: dict[str, object]
    if args.control_type == "custom" or args.control_fn:
        if not args.control_fn:
            raise ValueError("--control-type custom requires --control-fn.")
        control_fn = load_control_fn(args.control_fn)
        control_meta = {"type": "custom_callable", "spec": args.control_fn}
    elif args.control_type == "sinusoidal":
        sin_cfg = RandomSinusoidalControlConfig(
            amplitude=args.sin_amplitude,
            frequency_range=(args.sin_freq_range[0], args.sin_freq_range[1]),
            phase_range=(args.sin_phase_range[0], args.sin_phase_range[1]),
            bias=args.sin_bias,
        )
        control_fn, control_meta = make_random_sinusoidal_control_fn(
            num_trajectories=args.num_trajectories,
            config=sin_cfg,
            seed=args.seed,
        )
    else:
        control_fn, control_meta = make_constant_control_fn(
            num_trajectories=args.num_trajectories,
            value=args.constant_value,
        )

    payload = generate_dataset(
        num_trajectories=args.num_trajectories,
        horizon=args.horizon,
        dt=args.dt,
        seed=args.seed,
        control_fn=control_fn,
    )
    payload["control"] = control_meta
    save_dataset(payload, args.output)

    states = payload["states"]
    controls = payload["controls"]
    times = payload["times"]

    print(f"saved={args.output}")
    print(f"control={control_meta}")
    print(
        f"states_shape={tuple(states.shape)} controls_shape={tuple(controls.shape)} "
        f"time_steps={times.shape[0]} state_dim={states.shape[-1]}"
    )

    if args.plot:
        plot_stream_and_trajectories(
            states=states,
            config=config_from_payload(payload),
            title="Generated controlled trajectories over stream field",
            max_lines=args.max_lines,
            save_path=Path(args.image_dir) / "generated_data.pdf",
        )


if __name__ == "__main__":
    main()
