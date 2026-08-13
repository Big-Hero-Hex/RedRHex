"""Single allowlisted runner factory shared by train, play, and evaluation."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from packaging.version import Version

from .runners import SensorDistillationRunnerV2, SensorOnPolicyRunnerV2, VersionedTeacherRunnerV2


RSL_RL_V2_MINIMUM = Version("3.1.2")
RSL_RL_V2_MAXIMUM = Version("3.2")


@dataclass(frozen=True)
class RunnerCapabilitiesV2:
    v2: bool
    strict_checkpoint: bool
    sensor_history_input: bool
    asymmetric_critic: bool
    teacher_student_mixture: bool


_CAPABILITIES = {
    "OnPolicyRunner": RunnerCapabilitiesV2(False, False, False, False, False),
    "DistillationRunner": RunnerCapabilitiesV2(False, False, False, False, False),
    "VersionedTeacherRunnerV2": RunnerCapabilitiesV2(True, True, False, True, False),
    "SensorDistillationRunnerV2": RunnerCapabilitiesV2(True, True, True, False, True),
    "SensorOnPolicyRunnerV2": RunnerCapabilitiesV2(True, True, True, True, False),
}
_V2_CLASSES = {
    "VersionedTeacherRunnerV2": VersionedTeacherRunnerV2,
    "SensorDistillationRunnerV2": SensorDistillationRunnerV2,
    "SensorOnPolicyRunnerV2": SensorOnPolicyRunnerV2,
}


def runner_capabilities(class_name: str | None = None) -> dict[str, Any]:
    """Return immutable-by-copy capability data for one or every runner."""

    if class_name is not None:
        try:
            return asdict(_CAPABILITIES[class_name])
        except KeyError as exc:
            raise ValueError(f"runner class is not allowlisted: {class_name!r}") from exc
    return {name: asdict(capabilities) for name, capabilities in _CAPABILITIES.items()}


def require_rsl_rl_v2_version() -> str:
    """Apply the narrow RSL version gate only when a V2 runner is selected."""

    try:
        installed = Version(importlib.metadata.version("rsl-rl-lib"))
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Sensor V2 requires rsl-rl-lib >=3.1.2,<3.2") from exc
    if not RSL_RL_V2_MINIMUM <= installed < RSL_RL_V2_MAXIMUM:
        raise RuntimeError(
            f"Sensor V2 requires rsl-rl-lib >=3.1.2,<3.2; installed version is {installed}"
        )
    return str(installed)


def create_runner(
    class_name: str,
    env: Any,
    config: dict[str, Any],
    *,
    log_dir: str | None,
    device: str,
    v2_backend_factories: Mapping[str, Callable[..., Any]] | None = None,
) -> Any:
    """Create only an explicitly allowlisted V1 or V2 runner.

    V1 behavior delegates to the same stock classes as before.  A V2 backend
    must be injected by the Isaac integration and advertise the capabilities
    checked by :mod:`runners`; this prevents accidental fallback to a
    shape-compatible V1 runner.
    """

    if class_name not in _CAPABILITIES:
        raise ValueError(f"runner class is not allowlisted: {class_name!r}")
    if class_name == "OnPolicyRunner":
        from rsl_rl.runners import OnPolicyRunner

        return OnPolicyRunner(env, config, log_dir=log_dir, device=device)
    if class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        return DistillationRunner(env, config, log_dir=log_dir, device=device)

    require_rsl_rl_v2_version()
    backend_factory = (v2_backend_factories or {}).get(class_name)
    return _V2_CLASSES[class_name](
        env,
        config,
        log_dir=log_dir,
        device=device,
        backend_factory=backend_factory,
    )
