from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import unquote, urlparse


SOURCE_IDENTITY_SCHEMA_VERSION = "redrhex.autopilot.source-identity.v1"
DEPENDENCY_MANIFEST_SCHEMA_VERSION = "redrhex.autopilot.dependency-manifest.v1"

# These files are the in-repository inputs that can change how an Autopilot
# trial is compiled, launched, evaluated, or ranked. Keep the list explicit so
# an omitted or newly introduced input is a reviewable contract change.
AUTOPILOT_CODE_IDENTITY_PATHS: tuple[str, ...] = (
    "RedRhex.usd",
    "source/RedRhex/pyproject.toml",
    "tools/training_panel/training_panel/autopilot_identity.py",
    "tools/training_panel/training_panel/autopilot.py",
    "tools/training_panel/training_panel/autopilot_service.py",
    "tools/training_panel/training_panel/autopilot_store.py",
    "tools/training_panel/training_panel/commands.py",
    "tools/training_panel/training_panel/config.py",
    "tools/training_panel/training_panel/history.py",
    "tools/training_panel/training_panel/physics.py",
    "tools/training_panel/training_panel/processes.py",
    "tools/training_panel/training_panel/reward_overrides.py",
    "tools/training_panel/training_panel/rewards.py",
    "tools/training_panel/training_panel/terrain.py",
    "tools/sim2real/checkpoint_spring.py",
    "tools/sim2real/contracts.py",
    "tools/sim2real/isaac_profile.py",
    "tools/sim2real/physics_profile.py",
    "tools/sim2real/profile_measurements.py",
    "tools/sim2real/repo_binding.py",
    "tools/sim2real/traces.py",
    "scripts/rsl_rl/cli_args.py",
    "scripts/rsl_rl/train.py",
    "scripts/rsl_rl/eval_command_sweep.py",
    "scripts/rsl_rl/runner_factory.py",
    "source/RedRhex/RedRhex/__init__.py",
    "source/RedRhex/RedRhex/tasks/__init__.py",
    "source/RedRhex/RedRhex/tasks/direct/__init__.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/__init__.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/abad_target_mapping.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/__init__.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/agents/rsl_rl_ppo_cfg.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_symmetry.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/target_delay.py",
    "source/RedRhex/RedRhex/tasks/direct/redrhex/torsion_spring.py",
)

CONFIG_IDENTITY_PATH = (
    "source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py"
)
DEPENDENCY_SOURCE_PATHS: tuple[str, ...] = ("source/RedRhex/pyproject.toml",)

# The evaluator imports or relies directly on these distributions. Every entry
# is required; missing metadata fails campaign identity resolution.
DEPENDENCY_DISTRIBUTIONS: tuple[str, ...] = (
    "gymnasium",
    "hydra-core",
    "isaaclab",
    "isaaclab-assets",
    "isaaclab-rl",
    "isaaclab-tasks",
    "redrhex",
    "rsl-rl-lib",
)

# Torch and NumPy are injected by the Isaac Sim extension loader and therefore
# do not have reliable importlib metadata in the configured Python before Kit
# starts. Bind their authoritative bundled METADATA files together with the
# simulator, Kit application, and PhysX build manifests instead.
SIMULATOR_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("isaac_sim", "VERSION"),
    ("isaac_app", "apps/isaacsim.exp.base.python.kit"),
    ("kit_app", "kit/apps/omni.app.full.kit"),
    ("kit_core", "kit/kernel/config/kit-core.json"),
    (
        "numpy",
        "extscache/omni.kit.pip_archive-*/pip_prebundle/numpy-*.dist-info/METADATA",
    ),
    ("torch", "exts/omni.isaac.ml_archive/pip_prebundle/torch-*.dist-info/METADATA"),
    ("physx", "extscache/omni.physics.physx-*/config/extension.toml"),
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("identity source paths must be non-empty repository-relative paths")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"identity source escapes the repository: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"identity source is missing: {resolved}")
    return resolved


def build_source_manifest(
    root: str | Path,
    relative_paths: Sequence[str] = AUTOPILOT_CODE_IDENTITY_PATHS,
) -> dict[str, object]:
    root_path = Path(root)
    paths = tuple(relative_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("identity source paths must be unique")
    files = [
        {"path": relative, "sha256": sha256_file(_resolved_source(root_path, relative))}
        for relative in paths
    ]
    return {
        "schema_version": SOURCE_IDENTITY_SCHEMA_VERSION,
        "files": files,
    }


def source_manifest_sha256(
    root: str | Path,
    relative_paths: Sequence[str] = AUTOPILOT_CODE_IDENTITY_PATHS,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(build_source_manifest(root, relative_paths))
    ).hexdigest()


def source_code_identities(root: str | Path) -> dict[str, str]:
    root_path = Path(root)
    return {
        "code": source_manifest_sha256(root_path),
        "config": sha256_file(_resolved_source(root_path, CONFIG_IDENTITY_PATH)),
    }


def build_simulator_manifest(simulator_root: str | Path) -> dict[str, object]:
    root = Path(simulator_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Isaac Sim root is missing: {root}")
    components: list[dict[str, str]] = []
    for name, pattern in SIMULATOR_COMPONENT_PATTERNS:
        matches = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
        if len(matches) != 1:
            raise RuntimeError(
                f"Isaac Sim dependency {name} must resolve exactly once; "
                f"found {len(matches)} for {pattern}"
            )
        path = matches[0]
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Isaac Sim dependency escapes its root: {path}") from exc
        components.append(
            {"name": name, "path": relative.as_posix(), "sha256": sha256_file(path)}
        )
    return {"components": components}


def _run_identity_command(args: Sequence[str], *, cwd: Path) -> bytes:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"dependency source identity command failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise RuntimeError(
            "dependency source identity command failed"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _git_worktree_identity(origin: Path) -> str | None:
    try:
        git_root_raw = _run_identity_command(
            ("git", "rev-parse", "--show-toplevel"), cwd=origin
        )
        git_root = Path(git_root_raw.decode("utf-8").strip()).resolve()
        scope = origin.relative_to(git_root)
        scope_arg = scope.as_posix() if scope.parts else "."
        treeish = "HEAD^{tree}" if not scope.parts else f"HEAD:{scope_arg}"
        source_tree = _run_identity_command(
            ("git", "rev-parse", treeish), cwd=git_root
        ).strip()
        diff = _run_identity_command(
            (
                "git", "diff", "--no-ext-diff", "--binary", "HEAD", "--",
                scope_arg,
            ),
            cwd=git_root,
        )
        untracked_raw = _run_identity_command(
            (
                "git", "ls-files", "--others", "--exclude-standard", "-z",
                "--", scope_arg,
            ),
            cwd=git_root,
        )
    except (RuntimeError, UnicodeError, OSError, ValueError):
        return None
    untracked: list[dict[str, str]] = []
    for raw in sorted(part for part in untracked_raw.split(b"\0") if part):
        try:
            relative = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("editable dependency has a non-UTF-8 untracked path") from exc
        path = (git_root / relative).resolve()
        try:
            path.relative_to(git_root)
        except ValueError as exc:
            raise RuntimeError("editable dependency path escapes its Git worktree") from exc
        if path.is_symlink():
            value = hashlib.sha256(
                ("symlink:" + str(path.readlink())).encode("utf-8")
            ).hexdigest()
        elif path.is_file():
            value = sha256_file(path)
        else:
            value = hashlib.sha256(b"missing").hexdigest()
        untracked.append({"path": relative, "sha256": value})
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "source_tree": source_tree.decode("ascii"),
                "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "untracked": untracked,
            }
        )
    ).hexdigest()


_EDITABLE_SOURCE_SUFFIXES = {
    ".cfg", ".ini", ".json", ".kit", ".py", ".pyi", ".so", ".toml",
    ".usd", ".yaml", ".yml",
}


def _editable_tree_identity(origin: Path) -> str:
    if not origin.is_dir():
        raise RuntimeError("editable dependency origin is unavailable")
    files: list[dict[str, str]] = []
    ignored_parts = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
    for path in sorted(origin.rglob("*")):
        try:
            relative = path.relative_to(origin)
        except ValueError as exc:
            raise RuntimeError("editable dependency path escapes its origin") from exc
        if any(part in ignored_parts for part in relative.parts):
            continue
        if path.is_symlink():
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(
                        ("symlink:" + str(path.readlink())).encode("utf-8")
                    ).hexdigest(),
                }
            )
        elif path.is_file() and path.suffix.lower() in _EDITABLE_SOURCE_SUFFIXES:
            files.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    if not files:
        raise RuntimeError("editable dependency has no identity-bearing source files")
    return hashlib.sha256(canonical_json_bytes(files)).hexdigest()


def _installed_distribution_identity(
    name: str,
    version: str,
    *,
    distribution_resolver: Callable[[str], importlib_metadata.Distribution],
    editable_cache: dict[Path, tuple[str, str]],
) -> dict[str, str]:
    try:
        distribution = distribution_resolver(name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required dependency distribution is missing: {name}") from exc
    if str(distribution.version).strip() != version:
        raise RuntimeError(f"dependency metadata version mismatch: {name}")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"dependency direct_url metadata is malformed: {name}") from exc
        editable = bool(
            isinstance(direct_url, dict)
            and isinstance(direct_url.get("dir_info"), dict)
            and direct_url["dir_info"].get("editable") is True
        )
        if editable:
            parsed = urlparse(str(direct_url.get("url") or ""))
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise RuntimeError(f"editable dependency origin is not a local file URL: {name}")
            origin = Path(unquote(parsed.path)).resolve()
            cached = editable_cache.get(origin)
            if cached is None:
                git_identity = _git_worktree_identity(origin)
                cached = (
                    ("editable_git", git_identity)
                    if git_identity is not None
                    else ("editable_tree", _editable_tree_identity(origin))
                )
                editable_cache[origin] = cached
            install_kind, source_sha = cached
            content_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "direct_url_sha256": hashlib.sha256(
                            direct_url_text.encode("utf-8")
                        ).hexdigest(),
                        "source_sha256": source_sha,
                    }
                )
            ).hexdigest()
            return {"install_kind": install_kind, "content_sha256": content_sha}

    entries: list[dict[str, str]] = []
    distribution_files = distribution.files
    if not distribution_files:
        raise RuntimeError(f"dependency installed-file metadata is unavailable: {name}")
    for relative_file in sorted(distribution_files, key=lambda item: str(item)):
        path = Path(distribution.locate_file(relative_file))
        if path.is_symlink():
            digest = hashlib.sha256(
                ("symlink:" + str(path.readlink())).encode("utf-8")
            ).hexdigest()
        elif path.is_file():
            digest = sha256_file(path)
        else:
            raise RuntimeError(f"dependency installed file is missing: {name}")
        entries.append({"path": str(relative_file), "sha256": digest})
    return {
        "install_kind": "installed_files",
        "content_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def build_dependency_manifest(
    root: str | Path,
    *,
    simulator_root: str | Path,
    version_resolver: Callable[[str], str] = importlib_metadata.version,
    python_version: Sequence[int] | None = None,
    implementation: str | None = None,
    distribution_resolver: Callable[[str], importlib_metadata.Distribution] = importlib_metadata.distribution,
    distribution_identity_resolver: Callable[[str, str], Mapping[str, str]] | None = None,
) -> dict[str, object]:
    version_info = tuple(sys.version_info[:3] if python_version is None else python_version)
    if len(version_info) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in version_info):
        raise ValueError("python_version must contain exactly three integers")
    names = DEPENDENCY_DISTRIBUTIONS

    resolved_distributions: list[dict[str, str]] = []
    editable_cache: dict[Path, tuple[str, str]] = {}
    for name in sorted(names, key=str.casefold):
        try:
            version = version_resolver(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required dependency distribution is missing: {name}"
            ) from exc
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError(f"dependency version is unavailable: {name}")
        resolved_version = version.strip()
        identity = (
            dict(distribution_identity_resolver(name, resolved_version))
            if distribution_identity_resolver is not None
            else _installed_distribution_identity(
                name,
                resolved_version,
                distribution_resolver=distribution_resolver,
                editable_cache=editable_cache,
            )
        )
        if (
            set(identity) != {"install_kind", "content_sha256"}
            or identity.get("install_kind") not in {
                "editable_git", "editable_tree", "installed_files"
            }
            or re.fullmatch(r"[0-9a-f]{64}", str(identity.get("content_sha256") or "")) is None
        ):
            raise RuntimeError(f"dependency content identity is invalid: {name}")
        resolved_distributions.append(
            {"name": name, "version": resolved_version, **identity}
        )

    root_path = Path(root)
    source_files = [
        {"path": relative, "sha256": sha256_file(_resolved_source(root_path, relative))}
        for relative in DEPENDENCY_SOURCE_PATHS
    ]
    return {
        "schema_version": DEPENDENCY_MANIFEST_SCHEMA_VERSION,
        "python": {
            "implementation": implementation or platform.python_implementation(),
            "version": ".".join(str(value) for value in version_info),
        },
        "distributions": resolved_distributions,
        "simulator": build_simulator_manifest(simulator_root),
        "source_files": source_files,
    }


def dependency_manifest_sha256(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(manifest))).hexdigest()


def _validate_dependency_manifest(
    root: Path,
    simulator_root: Path,
    manifest: object,
) -> dict[str, object]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "python", "distributions", "simulator", "source_files"
    }:
        raise RuntimeError("dependency interpreter returned an invalid manifest schema")
    if manifest.get("schema_version") != DEPENDENCY_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("dependency interpreter returned an unsupported manifest")
    python = manifest.get("python")
    if (
        not isinstance(python, dict)
        or set(python) != {"implementation", "version"}
        or not isinstance(python.get("implementation"), str)
        or not python["implementation"]
        or not isinstance(python.get("version"), str)
        or re.fullmatch(r"\d+\.\d+\.\d+", python["version"]) is None
    ):
        raise RuntimeError("dependency interpreter returned invalid Python identity")
    distributions = manifest.get("distributions")
    if not isinstance(distributions, list):
        raise RuntimeError("dependency interpreter returned invalid distribution identity")
    names: list[str] = []
    for item in distributions:
        if not isinstance(item, dict) or set(item) != {
            "name", "version", "install_kind", "content_sha256"
        }:
            raise RuntimeError("dependency interpreter returned invalid distribution identity")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("dependency interpreter returned invalid distribution identity")
        if not isinstance(item.get("version"), str) or not item["version"]:
            raise RuntimeError("dependency interpreter returned invalid distribution version")
        if item.get("install_kind") not in {
            "editable_git", "editable_tree", "installed_files"
        } or re.fullmatch(r"[0-9a-f]{64}", str(item.get("content_sha256") or "")) is None:
            raise RuntimeError("dependency interpreter returned invalid distribution content")
        names.append(name)
    if names != sorted(DEPENDENCY_DISTRIBUTIONS, key=str.casefold):
        raise RuntimeError("dependency interpreter returned an incomplete distribution manifest")
    simulator = manifest.get("simulator")
    if not isinstance(simulator, dict):
        raise RuntimeError("dependency interpreter returned invalid simulator identity")
    expected_simulator = build_simulator_manifest(simulator_root)
    if simulator != expected_simulator:
        raise RuntimeError("dependency interpreter returned mismatched simulator identity")
    expected_sources = [
        {"path": relative, "sha256": sha256_file(_resolved_source(root, relative))}
        for relative in DEPENDENCY_SOURCE_PATHS
    ]
    if manifest.get("source_files") != expected_sources:
        raise RuntimeError("dependency interpreter returned mismatched dependency sources")
    return manifest


def dependency_manifest_for_python(
    root: str | Path,
    python_executable: str | Path,
    *,
    simulator_root: str | Path,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Resolve dependency versions in the interpreter that will run Isaac."""

    root_path = Path(root).resolve()
    executable = Path(python_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"dependency interpreter is missing: {executable}")
    resolved_simulator_root = Path(simulator_root).resolve()
    try:
        if executable.samefile(Path(sys.executable).resolve()):
            return _validate_dependency_manifest(
                root_path,
                resolved_simulator_root,
                build_dependency_manifest(
                    root_path, simulator_root=resolved_simulator_root
                ),
            )
    except OSError:
        pass
    probe = (
        "import sys;"
        "from pathlib import Path;"
        "root=Path(sys.argv[1]).resolve();"
        "sys.path.insert(0,str(root));"
        "from tools.training_panel.training_panel.autopilot_identity import "
        "build_dependency_manifest,canonical_json_bytes;"
        "simulator_root=Path(sys.argv[2]).resolve();"
        "manifest=build_dependency_manifest(root,simulator_root=simulator_root);"
        "sys.stdout.buffer.write(canonical_json_bytes(manifest))"
    )
    try:
        completed = subprocess.run(
            [
                str(executable), "-c", probe, str(root_path),
                str(resolved_simulator_root),
            ],
            cwd=str(root_path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"dependency interpreter probe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise RuntimeError(
            "dependency interpreter probe failed"
            + (f": {detail}" if detail else "")
        )
    try:
        parsed = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("dependency interpreter returned malformed JSON") from exc
    return _validate_dependency_manifest(root_path, resolved_simulator_root, parsed)


def runtime_source_identities(
    root: str | Path,
    *,
    simulator_root: str | Path,
    version_resolver: Callable[[str], str] = importlib_metadata.version,
    python_version: Sequence[int] | None = None,
    implementation: str | None = None,
    python_executable: str | Path | None = None,
    distribution_resolver: Callable[[str], importlib_metadata.Distribution] = importlib_metadata.distribution,
    distribution_identity_resolver: Callable[[str, str], Mapping[str, str]] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    root_path = Path(root)
    if python_executable is not None:
        if python_version is not None or implementation is not None:
            raise ValueError(
                "python_executable cannot be combined with an injected Python identity"
            )
        dependency_manifest = dependency_manifest_for_python(
            root_path,
            python_executable,
            simulator_root=simulator_root,
        )
    else:
        dependency_manifest = build_dependency_manifest(
            root_path,
            simulator_root=simulator_root,
            version_resolver=version_resolver,
            python_version=python_version,
            implementation=implementation,
            distribution_resolver=distribution_resolver,
            distribution_identity_resolver=distribution_identity_resolver,
        )
    return (
        {
            **source_code_identities(root_path),
            "dependency": dependency_manifest_sha256(dependency_manifest),
        },
        dependency_manifest,
    )
