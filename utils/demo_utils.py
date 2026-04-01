# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import time
from typing import Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_PATH = os.path.join(_REPO_ROOT, "output")
CKPT_PATH = os.path.join(_REPO_ROOT, "ckpts")
SAMPLE_DATA_PATH = os.path.join(_REPO_ROOT, "sample_data")


class CudaTimer:
    """
    A CUDA-aware timer that uses torch.cuda.Event for accurate GPU timing.

    For CUDA devices, uses cuda events which measure time on the GPU directly.
    For CPU/MPS devices, falls back to time.time().

    Example usage:
        timer = CudaTimer(device="cuda")

        # Manual start/stop style
        timer.start("preprocessing")
        preprocess(data)
        elapsed_ms = timer.stop("preprocessing")

        # Context manager style
        with timer("inference"):
            model(input)
        print(f"Inference took {timer.get_ms('inference'):.1f}ms")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the timer.

        Args:
            device: The device to time. If "cuda" and CUDA is available,
                   uses CUDA events for precise GPU timing.
        """
        self.device = device
        self.use_cuda_events = device == "cuda" and torch.cuda.is_available()
        self._timers: dict = {}
        self._current_name: Optional[str] = None

    def start(self, name: str = "default") -> "CudaTimer":
        """
        Start timing a named operation.

        Args:
            name: Identifier for this timing operation

        Returns:
            self, for method chaining
        """
        if self.use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self._timers[name] = {"start": start_event, "end": None, "elapsed_ms": None}
        else:
            self._timers[name] = {"start": time.time(), "end": None, "elapsed_ms": None}
        return self

    def stop(self, name: str = "default") -> float:
        """
        Stop timing a named operation and return elapsed time in milliseconds.

        Args:
            name: Identifier for the timing operation to stop

        Returns:
            Elapsed time in milliseconds
        """
        if name not in self._timers:
            raise ValueError(f"Timer '{name}' was never started")

        timer_data = self._timers[name]

        if self.use_cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            end_event.synchronize()
            elapsed_ms = timer_data["start"].elapsed_time(end_event)
            timer_data["end"] = end_event
        else:
            end_time = time.time()
            elapsed_ms = (end_time - timer_data["start"]) * 1000
            timer_data["end"] = end_time

        timer_data["elapsed_ms"] = elapsed_ms
        return elapsed_ms

    def get_ms(self, name: str = "default") -> float:
        """
        Get the elapsed time in milliseconds for a completed timing operation.

        Args:
            name: Identifier for the timing operation

        Returns:
            Elapsed time in milliseconds
        """
        if name not in self._timers:
            raise ValueError(f"Timer '{name}' was never started")
        if self._timers[name]["elapsed_ms"] is None:
            raise ValueError(f"Timer '{name}' was never stopped")
        return self._timers[name]["elapsed_ms"]

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset timer(s).

        Args:
            name: If provided, reset only this timer. Otherwise reset all timers.
        """
        if name is not None:
            if name in self._timers:
                del self._timers[name]
        else:
            self._timers.clear()

    def __call__(self, name: str = "default") -> "CudaTimer":
        """
        Prepare for use as context manager with given name.

        Args:
            name: Identifier for this timing operation

        Returns:
            self, for use with 'with' statement
        """
        self._current_name = name
        return self

    def __enter__(self) -> "CudaTimer":
        """Start timing when entering context."""
        name = self._current_name or "default"
        self.start(name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing when exiting context."""
        name = self._current_name or "default"
        self.stop(name)
        self._current_name = None


DEFAULT_SEQ = os.path.expanduser("~/boxy_data/anytable11")
