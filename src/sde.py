"""Base class for controlled stochastic differential equations."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
from torch import Tensor


def zero_order_hold(
    times: Tensor,
    values: Tensor,
    t: Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Zero-order hold lookup: return the control active at time *t*.

    Given interval boundaries *times* ``(K+1,)`` and stepwise-constant
    *values* ``(..., K, control_dim)``, return ``values[..., k, :]``
    where ``times[k] <= t < times[k+1]``.
    """
    device = device or values.device
    dtype = dtype or values.dtype

    times = times.to(device=device, dtype=dtype)
    values = values.to(device=device, dtype=dtype)

    t_scalar = t.to(device=device, dtype=dtype).squeeze()
    t_clamped = torch.clamp(t_scalar, times[0], times[-1])

    idx = int(torch.clamp(
        torch.searchsorted(times, t_clamped, right=True) - 1,
        0,
        values.shape[-2] - 1,
    ).item())

    return values[..., idx, :]


class ControlledSDE(nn.Module):
    """Base SDE with stepwise-constant control schedule management.

    Subclasses must implement :meth:`f` (drift) and :meth:`g` (diffusion),
    using :meth:`_lookup_control` to obtain the active control value at any
    time *t* during integration.

    Compatible with ``torchsde.sdeint`` out of the box.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, control_dim: int = 1) -> None:
        super().__init__()
        self.control_dim = control_dim
        self._control_times: Tensor | None = None
        self._control_values: Tensor | None = None

    # -- control schedule management --

    def set_control_trajectory(self, times: Tensor, controls: Tensor) -> None:
        """Register stepwise-constant controls for SDE integration.

        Args:
            times: ``(K+1,)`` interval boundaries.
            controls: ``(K, control_dim)`` or ``(batch, K, control_dim)``.
        """
        self._control_times = times
        self._control_values = controls

    def clear_control_trajectory(self) -> None:
        self._control_times = None
        self._control_values = None

    @contextmanager
    def controlled(
        self, times: Tensor, controls: Tensor,
    ) -> Generator[ControlledSDE, None, None]:
        """Context manager: set controls for the duration of an integration."""
        self.set_control_trajectory(times, controls)
        try:
            yield self
        finally:
            self.clear_control_trajectory()

    def _lookup_control(self, t: Tensor, y: Tensor) -> Tensor:
        """Return the active control at time *t* via zero-order hold."""
        if self._control_times is None or self._control_values is None:
            return torch.zeros(
                (y.shape[0], self.control_dim),
                device=y.device,
                dtype=y.dtype,
            )
        return zero_order_hold(
            self._control_times, self._control_values, t,
            device=y.device, dtype=y.dtype,
        )

    # -- SDE interface (subclasses must override) --

    @abstractmethod
    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Drift coefficient."""

    @abstractmethod
    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Diffusion coefficient."""
