"""Capability-checked runner adapters for the Isaac/RSL integration boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class V2RunnerIntegrationError(RuntimeError):
    """Raised when an injected backend does not implement the V2 contract."""


class _CapabilityCheckedRunnerV2:
    required_capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        env: Any,
        config: dict[str, Any],
        *,
        log_dir: str | None,
        device: str,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        if backend_factory is None:
            raise V2RunnerIntegrationError(
                f"{type(self).__name__} requires an Isaac/RSL backend factory; "
                "using a stock V1 runner would violate the V2 policy and checkpoint contracts"
            )
        backend = backend_factory(env, config, log_dir=log_dir, device=device)
        actual = frozenset(getattr(backend, "sensor_v2_capabilities", ()))
        missing = self.required_capabilities - actual
        if missing:
            raise V2RunnerIntegrationError(
                f"{type(self).__name__} backend is missing V2 capabilities: {sorted(missing)}"
            )
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.learn(*args, **kwargs)

    def load(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.load(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.save(*args, **kwargs)


class VersionedTeacherRunnerV2(_CapabilityCheckedRunnerV2):
    required_capabilities = frozenset(
        {"strict_checkpoint_v2", "teacher_rollout_v2", "versioned_provenance_v2"}
    )


class SensorDistillationRunnerV2(_CapabilityCheckedRunnerV2):
    required_capabilities = frozenset(
        {
            "strict_checkpoint_v2",
            "two_input_sensor_actor_v2",
            "three_action_streams_v2",
            "next_frame_terminal_mask_v2",
        }
    )


class SensorOnPolicyRunnerV2(_CapabilityCheckedRunnerV2):
    required_capabilities = frozenset(
        {
            "strict_checkpoint_v2",
            "two_input_sensor_actor_v2",
            "asymmetric_critic_v2",
            "distilled_actor_exact_bootstrap_v2",
        }
    )
