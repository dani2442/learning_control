# Learning Control with Neural Dynamics

Learn a controlled nonlinear flow from trajectory data, then solve and compare open-loop optimal control on:
1. the known physical dynamics, and
2. the learned neural surrogate.

Built with `PyTorch`.

## System We Model

State and control:
- `x = [x, y] in R^2`
- `u in R`

Controlled vortex dynamics (`src/dataset.py`):

```math
\dot{x} = U + \sum_{i=1}^{3}\Gamma_i\frac{2y}{(x-p_i)^2 + y^2 + \varepsilon} + k_x u
```

```math
\dot{y} = \sum_{i=1}^{3}-\Gamma_i\frac{2(x-p_i)}{(x-p_i)^2 + y^2 + \varepsilon} + k_y u
```

Default parameters:
- `U = 1.0`
- `p = (-2, 0, 2)`
- `Gamma = (1, -1, 1)`
- `k_x = 0.05`, `k_y = 0.5`
- `epsilon = 1e-3`

Training data is generated with deterministic Euler integration:

```math
X_{k+1} = X_k + \Delta t\, f(X_k, u_k)
```

Real-system noise is set to zero by default (`diffusion = 0.0`).

The learned model is drift-only (no learned diffusion):

```math
dX_t = \big(f_\theta(X_t, t) + B u_t\big)\,dt
```

## Installation

Requires `Python >= 3.13` (from `pyproject.toml`).

### Option A: uv (recommended)
```bash
uv sync
```

### Option B: pip + venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

### 1) Generate data
Random sinusoidal control (`u_control(t)` with random frequency and phase):
```bash
.venv/bin/python examples/generate_data.py \
  --output data/controlled_vortex.pt \
  --num-trajectories 512 \
  --horizon 4.0 \
  --dt 0.05 \
  --control-type sinusoidal \
  --sin-amplitude 0.6 \
  --sin-freq-range 0.4 2.0 \
  --sin-phase-range 0.0 6.283185307
```

Constant control (`u_control(t) = c`, modifiable with `--constant-value`):
```bash
.venv/bin/python examples/generate_data.py \
  --output data/controlled_vortex_constant.pt \
  --num-trajectories 512 \
  --horizon 4.0 \
  --dt 0.05 \
  --control-type constant \
  --constant-value 0.2
```

### 2) Train neural dynamics model
```bash
.venv/bin/python examples/train_model.py \
  --dataset data/controlled_vortex.pt \
  --checkpoint data/neural_sde.pt \
  --epochs 250 \
  --batch-size 64 \
  --hidden-dim 128 \
  --val-ratio 0.2
```

### 3) Compare optimal control transfer
```bash
.venv/bin/python examples/optimal_control.py \
  --dataset data/controlled_vortex.pt \
  --checkpoint data/neural_sde.pt \
  --x0 -3.0 1.2 \
  --target 2.5 -0.8 \
  --horizon-steps 80 \
  --iters 500
```

Printed metrics are terminal L2 errors for:
- optimize on real -> evaluate on real
- optimize on real -> evaluate on learned
- optimize on learned -> evaluate on learned
- optimize on learned -> evaluate on real

## Minimal Python Example

```python
import torch
from torch.utils.data import DataLoader

from src.dataset import ControlledTrajectoryDataset, generate_dataset
from src.model import ControlledNeuralSDE, TrainingConfig, train_neural_sde

# 1) Generate a small dataset in memory.
payload = generate_dataset(num_trajectories=128, horizon=2.0, dt=0.05, seed=7)
dataset = ControlledTrajectoryDataset(payload)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2) Train a controlled drift-only model.
model = ControlledNeuralSDE(hidden_dim=64)
cfg = TrainingConfig(epochs=10, lr=1e-3, solver_dt=0.05)
history = train_neural_sde(
    model=model,
    dataloader=loader,
    times=payload["times"],
    config=cfg,
    device="cpu",
)

print("final_train_loss:", history["train_loss"][-1])
```

## Repository Layout

- `src/dataset.py`: dynamics, control policies, rollout, dataset IO
- `src/model.py`: controlled neural dynamics model, training, checkpoint IO
- `src/control.py`: open-loop optimal control on real/learned dynamics
- `src/visualization.py`: stream plots, trajectory comparison, error maps
- `examples/generate_data.py`: dataset generation CLI
- `examples/train_model.py`: model training CLI
- `examples/optimal_control.py`: transfer comparison CLI
- `examples/compute_control.py`: optimize on one selected dynamics model

## Notes

- `data/` artifacts are git-ignored.
- `--plot` is available on generation/control scripts for immediate visualization.
- `train_model.py` supports train/validation split via `--val-ratio`.
