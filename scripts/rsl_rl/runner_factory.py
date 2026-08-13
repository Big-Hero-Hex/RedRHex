"""Shared allowlisted runner construction for RedRHex RSL-RL scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunnerProtocol:
    class_name: str
    v2: bool
    strict_checkpoint: bool
    checkpoint_kind: str | None
    bootstrap_kind: str | None
    export_kind: str


_PROTOCOLS = {
    "OnPolicyRunner": RunnerProtocol(
        "OnPolicyRunner", False, False, None, None, "legacy"
    ),
    "DistillationRunner": RunnerProtocol(
        "DistillationRunner", False, False, None, None, "legacy"
    ),
    "VersionedTeacherRunnerV2": RunnerProtocol(
        "VersionedTeacherRunnerV2", True, True, "teacher_v2", None, "teacher_v2"
    ),
    "SensorDistillationRunnerV2": RunnerProtocol(
        "SensorDistillationRunnerV2",
        True,
        True,
        "student_distilled_v2",
        "teacher_v2",
        "student_v2",
    ),
    "SensorOnPolicyRunnerV2": RunnerProtocol(
        "SensorOnPolicyRunnerV2",
        True,
        True,
        "student_ppo_v2",
        "student_distilled_v2",
        "student_v2",
    ),
}


def runner_protocol(class_name: str) -> RunnerProtocol:
    try:
        return _PROTOCOLS[class_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported runner class {class_name!r}; allowed classes are {sorted(_PROTOCOLS)}"
        ) from exc


def create_runner(
    class_name: str,
    env: Any,
    config: dict[str, Any],
    *,
    log_dir: str | None,
    device: str,
) -> Any:
    """Construct a V1 or V2 runner through one explicit allowlist."""

    protocol = runner_protocol(class_name)
    if not protocol.v2:
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        cls = OnPolicyRunner if class_name == "OnPolicyRunner" else DistillationRunner
        return cls(env, config, log_dir=log_dir, device=device)

    from RedRhex.tasks.direct.redrhex.agents.sensor_v2.runner_factory import (
        create_runner as create_sensor_v2_runner,
    )
    from RedRhex.tasks.direct.redrhex.agents.sensor_v2.backends import (
        backend_factories_v2,
    )

    return create_sensor_v2_runner(
        class_name,
        env,
        config,
        log_dir=log_dir,
        device=device,
        v2_backend_factories=backend_factories_v2(),
    )


def get_exportable_actor(runner: Any, protocol: RunnerProtocol) -> Any:
    """Resolve an export graph without guessing policy ownership for V2."""

    if protocol.v2:
        candidate = getattr(runner, "get_exportable_actor", None)
        if candidate is None:
            candidate = getattr(runner, "exportable_actor", None)
        if candidate is None:
            raise RuntimeError(f"{protocol.class_name} does not expose a V2 exportable actor")
        return candidate() if callable(candidate) else candidate
    return runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic


def load_runner_checkpoint(
    runner: Any,
    checkpoint: str,
    *,
    device: str,
    protocol: RunnerProtocol,
    load_optimizer: bool = True,
    legacy_loader=None,
) -> None:
    """Keep compatibility fallback reachable only from V1 protocols."""

    if protocol.strict_checkpoint:
        try:
            runner.load(checkpoint, load_optimizer=load_optimizer, map_location=device)
        except TypeError:
            runner.load(checkpoint, load_optimizer=load_optimizer)
        return
    if legacy_loader is not None:
        legacy_loader(
            runner,
            checkpoint,
            device,
            load_optimizer=load_optimizer,
            allow_partial_policy=not load_optimizer,
        )
        return
    runner.load(checkpoint)
