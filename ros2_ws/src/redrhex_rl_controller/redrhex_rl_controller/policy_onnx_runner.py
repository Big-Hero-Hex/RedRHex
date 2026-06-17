from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class OnnxIOInfo:
    input_name: str
    input_shape: Sequence[object]
    input_type: str
    output_name: str
    output_shape: Sequence[object]
    output_type: str
    providers: list[str]


class PolicyONNXRunner:
    """Small ONNX Runtime wrapper with deployment-time safety checks."""

    def __init__(
        self,
        onnx_path: str,
        expected_obs_dim: int = 56,
        expected_action_dim: int = 12,
        use_cuda: bool = False,
        use_tensorrt: bool = False,
    ) -> None:
        self.onnx_path = os.path.expanduser(onnx_path)
        self.expected_obs_dim = int(expected_obs_dim)
        self.expected_action_dim = int(expected_action_dim)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX policy not found: {self.onnx_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. On Jetson install onnxruntime or onnxruntime-gpu first."
            ) from exc

        available = ort.get_available_providers()
        providers: list[str] = []
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if use_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.input_name = self.input.name
        self.output_name = self.output.name
        self.info = OnnxIOInfo(
            input_name=self.input.name,
            input_shape=list(self.input.shape),
            input_type=self.input.type,
            output_name=self.output.name,
            output_shape=list(self.output.shape),
            output_type=self.output.type,
            providers=list(self.session.get_providers()),
        )
        self._validate_static_io()

    def _validate_static_io(self) -> None:
        def dim_compatible(shape: Sequence[object], expected_last: int) -> bool:
            if len(shape) < 2:
                return False
            last = shape[-1]
            return last in (expected_last, "obs_dim", None) or isinstance(last, str)

        if not dim_compatible(self.info.input_shape, self.expected_obs_dim):
            raise ValueError(
                f"ONNX input shape {self.info.input_shape} is not compatible with [1,{self.expected_obs_dim}]"
            )
        if not dim_compatible(self.info.output_shape, self.expected_action_dim):
            raise ValueError(
                f"ONNX output shape {self.info.output_shape} is not compatible with [1,{self.expected_action_dim}]"
            )

    def run(self, observation: np.ndarray | Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape != (1, self.expected_obs_dim):
            raise ValueError(f"Expected observation shape (1,{self.expected_obs_dim}), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("Observation contains NaN or Inf")
        action = self.session.run([self.output_name], {self.input_name: obs})[0]
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape != (1, self.expected_action_dim):
            raise ValueError(f"Expected action shape (1,{self.expected_action_dim}), got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("ONNX action contains NaN or Inf")
        return action[0].copy()

    def zero_check(self) -> np.ndarray:
        return self.run(np.zeros((1, self.expected_obs_dim), dtype=np.float32))

    def describe(self) -> str:
        return (
            f"input={self.info.input_name} shape={list(self.info.input_shape)} type={self.info.input_type}\n"
            f"output={self.info.output_name} shape={list(self.info.output_shape)} type={self.info.output_type}\n"
            f"providers={self.info.providers}"
        )
