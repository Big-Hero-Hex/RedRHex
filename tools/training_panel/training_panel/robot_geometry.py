"""Robot layout for the Physics-page 3D viewer.

Parses the URDF that the deploy pipeline already treats as the canonical model and
emits a compact JSON layout keyed by the canonical leg index. The viewer needs
positions, not meshes, so only joint origins and axes are extracted.

Two deliberate choices:

* Leg identity is keyed by canonical index, never by label. The joint-name order in
  the contract is authoritative; ``_LEG_LABELS`` is only decoration on top of it.
* Every leg reports both its contract label and the position derived from the URDF.
  These disagree today on the right side (see ``label_audit``), so the viewer shows
  the true CAD position and flags the mismatch instead of silently picking one.
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from typing import Any

from .config import PanelPaths
from .deploy import FALLBACK_CONTRACT, _contract, default_mujoco_model_path
from .physics import FIELD_MAP, _LEG_LABELS


LAYOUT_VERSION = 1

# The CAD frame is Y-up. Isaac re-orients the articulation with quaternion
# (0.7071068, 0.7071068, 0, 0) -- a +90 deg rotation about X -- in
# source/RedRhex/RedRhex/tasks/direct/redrhex/redrhex_env_cfg.py. The viewer applies
# the same rotation so that "up" on screen matches the simulator, not the CAD file.
SIM_ROTATION_QUAT_WXYZ = (0.7071068, 0.7071068, 0.0, 0.0)

_ROW_NAMES = ("front", "middle", "rear")

# Spawn pose, used so the viewer shows the robot as the simulator starts it. The damper
# angle is deliberately NOT taken from here -- it is driven by the tunable
# passive_spring.damper_N.rest_position_rad field, whose schema default already matches.
_FALLBACK_INIT_MAIN_DRIVE_POS = [0.7853981634] * 3 + [-0.7853981634] * 3
_FALLBACK_INIT_ABAD_POS = [0.0] * 6

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_KEY: tuple[str, int, int] | None = None


def _floats(element: ET.Element | None, attribute: str, default: tuple[float, ...]) -> list[float]:
    if element is None:
        return list(default)
    raw = element.get(attribute)
    if not raw:
        return list(default)
    try:
        values = [float(part) for part in raw.split()]
    except ValueError:
        return list(default)
    return values if len(values) == len(default) else list(default)


def _urdf_name(canonical_joint: str) -> str:
    """Contract names use underscores; the URDF uses a space (``Revolute 15``)."""

    return canonical_joint.replace("_", " ", 1) if canonical_joint.startswith("Revolute_") else canonical_joint


def _joint_table(root: ET.Element) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if not name or parent is None or child is None:
            continue
        table[name] = {
            "name": name,
            "type": joint.get("type", "fixed"),
            "parent": parent.get("link", ""),
            "child": child.get("link", ""),
            "origin": _floats(joint.find("origin"), "xyz", (0.0, 0.0, 0.0)),
            "axis": _floats(joint.find("axis"), "xyz", (0.0, 0.0, 0.0)),
        }
    return table


def _contract_joint_names(paths: PanelPaths) -> dict[str, list[str]]:
    contract = _contract(paths)
    return {
        "main": list(getattr(contract, "MAIN_DRIVE_JOINT_NAMES", FALLBACK_CONTRACT.MAIN_DRIVE_JOINT_NAMES)),
        "abad": list(getattr(contract, "ABAD_JOINT_NAMES", FALLBACK_CONTRACT.ABAD_JOINT_NAMES)),
        "damper": list(getattr(contract, "DAMPER_JOINT_NAMES", FALLBACK_CONTRACT.DAMPER_JOINT_NAMES)),
    }


def _schema_default(key: str, fallback: float) -> float:
    """Inherited value for a field, so an un-overridden preset shows the real pose."""

    field = FIELD_MAP.get(key)
    if field is None or field.default is None:
        return fallback
    return float(field.default)


def _rest_defaults() -> list[float]:
    return [
        _schema_default(f"simulation_physics.passive_spring.damper_{index}.rest_position_rad", 0.0)
        for index in range(6)
    ]


def _contract_init_pose(paths: PanelPaths) -> dict[str, list[float]]:
    contract = _contract(paths)
    main = list(getattr(contract, "INIT_MAIN_DRIVE_POS", _FALLBACK_INIT_MAIN_DRIVE_POS))
    abad = list(getattr(contract, "INIT_ABAD_POS", _FALLBACK_INIT_ABAD_POS))
    return {
        "main": main if len(main) == 6 else list(_FALLBACK_INIT_MAIN_DRIVE_POS),
        "abad": abad if len(abad) == 6 else list(_FALLBACK_INIT_ABAD_POS),
        "damper": _rest_defaults(),
    }


def _nominal_com(root: ET.Element) -> list[float]:
    for link in root.findall("link"):
        if link.get("name") != "base_link":
            continue
        inertial = link.find("inertial")
        if inertial is not None:
            return _floats(inertial.find("origin"), "xyz", (0.0, 0.0, 0.0))
    return [0.0, 0.0, 0.0]


def _derive_positions(legs: list[dict[str, Any]], lateral_center: float) -> None:
    """Label each leg front/middle/rear from its own side's fore-aft ordering."""

    for leg in legs:
        leg["side"] = "right" if leg["mount"][2] > lateral_center else "left"
    for side in ("right", "left"):
        members = sorted(
            (leg for leg in legs if leg["side"] == side),
            key=lambda leg: leg["mount"][0],
        )
        for row_index, leg in enumerate(members):
            row = _ROW_NAMES[row_index] if row_index < len(_ROW_NAMES) else f"row{row_index}"
            leg["row"] = row
            leg["geometric_label"] = f"{leg['side'].capitalize()} {row}"


def build_layout(paths: PanelPaths) -> dict[str, Any]:
    """Return the viewer layout, falling back to a flat hexapod if the URDF is absent."""

    model_path = default_mujoco_model_path(paths)
    joint_names = _contract_joint_names(paths)
    init_pose = _contract_init_pose(paths)
    init_pose["damper"] = _rest_defaults()
    try:
        root = ET.parse(model_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return _fallback_layout(joint_names, init_pose, reason=str(exc))

    joints = _joint_table(root)
    by_child = {joint["child"]: joint for joint in joints.values()}

    legs: list[dict[str, Any]] = []
    for index in range(6):
        try:
            abad = joints[_urdf_name(joint_names["abad"][index])]
            main = joints[_urdf_name(joint_names["main"][index])]
            damper = joints[_urdf_name(joint_names["damper"][index])]
            mount_joint = by_child[abad["parent"]]
        except (KeyError, IndexError) as exc:
            return _fallback_layout(joint_names, init_pose, reason=f"leg {index} not resolvable: {exc}")
        # Chain must be connector -> abad -> motor holder -> main -> leg connect -> damper -> foot.
        if abad["child"] != main["parent"] or main["child"] != damper["parent"]:
            return _fallback_layout(joint_names, init_pose, reason=f"leg {index} chain is not connector/abad/main/damper")
        legs.append(
            {
                "index": index,
                "contract_label": _LEG_LABELS[index],
                "mount": mount_joint["origin"],
                "mount_link": abad["parent"],
                "leg_radius_m": abs(damper["origin"][0]),
                "joints": {
                    "abad": _joint_payload("abad", index, abad, init_pose["abad"][index]),
                    "main": _joint_payload("main", index, main, init_pose["main"][index]),
                    "damper": _joint_payload("damper", index, damper, init_pose["damper"][index]),
                },
            }
        )

    lateral = [leg["mount"][2] for leg in legs]
    lateral_center = (min(lateral) + max(lateral)) / 2.0
    _derive_positions(legs, lateral_center)

    fore_aft = [leg["mount"][0] for leg in legs]
    vertical = [leg["mount"][1] for leg in legs]
    return {
        "version": LAYOUT_VERSION,
        "source": "urdf",
        "model_path": str(model_path),
        "frame": {
            "forward_axis": "x",
            "up_axis": "y",
            "lateral_axis": "z",
            "sim_rotation_quat_wxyz": list(SIM_ROTATION_QUAT_WXYZ),
        },
        "body": {
            "fore_aft_span_m": [min(fore_aft), max(fore_aft)],
            "lateral_span_m": [min(lateral), max(lateral)],
            "vertical_m": sum(vertical) / len(vertical),
            "lateral_center_m": lateral_center,
            "nominal_com_m": _nominal_com(root),
        },
        "legs": legs,
        "label_audit": _label_audit(legs),
    }


def _joint_payload(group: str, index: int, joint: dict[str, Any], init_rad: float) -> dict[str, Any]:
    return {
        "canonical_id": f"{group}_{index}",
        "urdf_name": joint["name"],
        "origin_m": joint["origin"],
        "axis": joint["axis"],
        "init_rad": float(init_rad),
    }


def _label_audit(legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Report legs whose contract label disagrees with the URDF position.

    Kept as data rather than an exception: the canonical indices and the tripod
    grouping are correct, so training is unaffected, but an operator reading
    "Right front" on the Physics page needs to know which leg that really is.
    """

    mismatched = [
        {
            "index": leg["index"],
            "contract_label": leg["contract_label"],
            "geometric_label": leg["geometric_label"],
        }
        for leg in legs
        if leg["contract_label"].lower() != leg["geometric_label"].lower()
    ]
    return {"consistent": not mismatched, "mismatched": mismatched}


def _fallback_layout(joint_names: dict[str, list[str]], init_pose: dict[str, list[float]], reason: str) -> dict[str, Any]:
    """A plain 3x2 hexapod so the Physics page still renders without the mesh package."""

    legs: list[dict[str, Any]] = []
    for index in range(6):
        side_sign = 1.0 if index < 3 else -1.0
        row = index % 3
        mount = [0.22 * (row - 1), 0.0, 0.09 + side_sign * (0.09 + (0.08 if row == 1 else 0.0))]
        legs.append(
            {
                "index": index,
                "contract_label": _LEG_LABELS[index],
                "mount": mount,
                "mount_link": "",
                "leg_radius_m": 0.0845,
                "joints": {
                    group: {
                        "canonical_id": f"{group}_{index}",
                        "urdf_name": joint_names[group][index] if index < len(joint_names[group]) else "",
                        "origin_m": [0.0, 0.0, 0.0],
                        "axis": [1.0, 0.0, 0.0] if group == "abad" else [0.0, 0.0, 1.0],
                        "init_rad": float(init_pose.get(group, [0.0] * 6)[index]) if group in init_pose else 0.0,
                    }
                    for group in ("abad", "main", "damper")
                },
            }
        )
    _derive_positions(legs, 0.09)
    return {
        "version": LAYOUT_VERSION,
        "source": "fallback",
        "model_path": "",
        "fallback_reason": reason,
        "frame": {
            "forward_axis": "x",
            "up_axis": "y",
            "lateral_axis": "z",
            "sim_rotation_quat_wxyz": list(SIM_ROTATION_QUAT_WXYZ),
        },
        "body": {
            "fore_aft_span_m": [-0.22, 0.22],
            "lateral_span_m": [-0.08, 0.26],
            "vertical_m": 0.0,
            "lateral_center_m": 0.09,
            "nominal_com_m": [0.0, 0.0, 0.09],
        },
        "legs": legs,
        "label_audit": _label_audit(legs),
    }


def robot_geometry(paths: PanelPaths) -> dict[str, Any]:
    """Cached layout, invalidated when the URDF changes on disk."""

    global _CACHE, _CACHE_KEY
    model_path = default_mujoco_model_path(paths)
    try:
        stat = model_path.stat()
        key = (str(model_path), int(stat.st_mtime), int(stat.st_size))
    except OSError:
        key = (str(model_path), 0, 0)
    with _CACHE_LOCK:
        if _CACHE is not None and _CACHE_KEY == key:
            return _CACHE
        layout = build_layout(paths)
        _CACHE, _CACHE_KEY = layout, key
        return layout
