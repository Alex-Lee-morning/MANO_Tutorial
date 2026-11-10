"""MANO right-hand skeleton: joint order, parents, and a schematic forward kinematics.

Bone lengths here are a teaching sketch. They are not the official template.
Official vertices, skinning weights, and joint positions come from MANO_RIGHT.pkl
via official_mano.py.
"""

from __future__ import annotations

import numpy as np

# Same 16 joints as the released MANO / SMPL-H hand.
# Pinky comes before ring. The pose vector is not thumb-to-pinky order.
JOINT_NAMES = [
    "wrist",
    "index1",
    "index2",
    "index3",
    "middle1",
    "middle2",
    "middle3",
    "pinky1",
    "pinky2",
    "pinky3",
    "ring1",
    "ring2",
    "ring3",
    "thumb1",
    "thumb2",
    "thumb3",
]

# thumb1 is the carpometacarpal joint, not the metacarpophalangeal joint.
ANATOMICAL = {
    "wrist": "wrist",
    "index1": "index MCP",
    "index2": "index PIP",
    "index3": "index DIP",
    "middle1": "middle MCP",
    "middle2": "middle PIP",
    "middle3": "middle DIP",
    "pinky1": "pinky MCP",
    "pinky2": "pinky PIP",
    "pinky3": "pinky DIP",
    "ring1": "ring MCP",
    "ring2": "ring PIP",
    "ring3": "ring DIP",
    "thumb1": "thumb CMC",
    "thumb2": "thumb MCP",
    "thumb3": "thumb IP",
}

# Row 0 of kintree_table. The file often stores the root parent as unsigned 4294967295.
PARENTS = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14],
    dtype=np.int64,
)

FINGERS = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}

# Fingertips are mesh vertices, not kinematic joints.
# These ids are the MANO entries in smplx/vertex_ids.py.
TIP_VERTEX_IDS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 554,
    "pinky": 671,
}

# Teaching offsets in millimetres, parent-relative, rest pose.
# This frame is only for the sketch. The official template has its own orientation.
#   +X  thumb -> pinky
#   +Y  wrist -> fingertips
#   +Z  back of the hand -> palm
_REST_OFFSET = np.array(
    [
        [0.0, 0.0, 0.0],  # 0 wrist
        [-20.0, 88.0, 4.0],  # 1 index MCP
        [-2.0, 40.0, 0.0],  # 2 index PIP
        [-1.0, 24.0, 0.0],  # 3 index DIP
        [0.0, 92.0, 2.0],  # 4 middle MCP
        [0.0, 44.0, 0.0],  # 5 middle PIP
        [0.0, 26.0, 0.0],  # 6 middle DIP
        [36.0, 72.0, 0.0],  # 7 pinky MCP
        [2.0, 32.0, 0.0],  # 8 pinky PIP
        [1.0, 18.0, 0.0],  # 9 pinky DIP
        [18.0, 84.0, 2.0],  # 10 ring MCP
        [1.0, 40.0, 0.0],  # 11 ring PIP
        [1.0, 24.0, 0.0],  # 12 ring DIP
        [-28.0, 30.0, 14.0],  # 13 thumb CMC
        [-26.0, 16.0, 10.0],  # 14 thumb MCP
        [-20.0, 6.0, 6.0],  # 15 thumb IP
    ],
    dtype=np.float64,
)

# Teaching fingertips, hung on the last joint of each finger.
# They are still not MANO joints 16..20. MANO only has joints 0..15.
_TIP_OFFSET = {
    3: np.array([-1.0, 22.0, 0.0]),
    6: np.array([0.0, 24.0, 0.0]),
    9: np.array([1.0, 16.0, 0.0]),
    12: np.array([1.0, 22.0, 0.0]),
    15: np.array([-16.0, 2.0, 4.0]),
}

FINGER_COLOR = {
    "wrist": "#4b5563",
    "index": "#2563eb",
    "middle": "#059669",
    "ring": "#d97706",
    "pinky": "#db2777",
    "thumb": "#7c3aed",
}


def finger_of(joint_index: int) -> str:
    if joint_index == 0:
        return "wrist"
    for name, joints in FINGERS.items():
        if joint_index in joints:
            return name
    raise IndexError(joint_index)


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle to a 3x3 rotation. Every MANO joint pose is one of these."""
    r = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx, ky, kz = k
    K = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def zero_pose() -> np.ndarray:
    """(16, 3) axis-angle. Joint 0 is the global rotation. The other 15 are the fingers."""
    return np.zeros((16, 3), dtype=np.float64)


def pose_from_dict(rotations_deg: dict[str, tuple[float, float, float]]) -> np.ndarray:
    """Write local axis-angle by joint name. Angles are degrees.

    At rest, each joint frame is aligned with the model axes. It is not a
    bone-aligned frame. In this teaching skeleton, flexion of the four fingers
    is a positive rotation about local X, which swings the fingertip toward +Z.
    """
    pose = zero_pose()
    name_to_index = {name: i for i, name in enumerate(JOINT_NAMES)}
    for name, xyz_deg in rotations_deg.items():
        pose[name_to_index[name]] = np.deg2rad(np.asarray(xyz_deg, dtype=np.float64))
    return pose


def forward_kinematics(
    local_axis_angle: np.ndarray,
    rest_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parent rotations carry child bones. A joint's own rotation does not move it.

    Returns
        positions: (16, 3)
        orientations: (16, 3, 3) global rotations
    """
    local_axis_angle = np.asarray(local_axis_angle, dtype=np.float64).reshape(16, 3)
    offsets = _REST_OFFSET if rest_offset is None else np.asarray(rest_offset, dtype=np.float64)
    positions = np.zeros((16, 3), dtype=np.float64)
    orientations = np.zeros((16, 3, 3), dtype=np.float64)
    for index in range(16):
        local_rotation = rodrigues(local_axis_angle[index])
        parent = int(PARENTS[index])
        if parent < 0:
            orientations[index] = local_rotation
            positions[index] = offsets[index]
        else:
            orientations[index] = orientations[parent] @ local_rotation
            positions[index] = positions[parent] + orientations[parent] @ offsets[index]
    return positions, orientations


def fingertip_positions(
    positions: np.ndarray,
    orientations: np.ndarray,
) -> dict[str, np.ndarray]:
    tips = {}
    for joint_index, offset in _TIP_OFFSET.items():
        name = finger_of(joint_index)
        tips[name] = positions[joint_index] + orientations[joint_index] @ offset
    return tips


def children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {i: [] for i in range(16)}
    for index, parent in enumerate(PARENTS):
        if parent >= 0:
            children[int(parent)].append(index)
    return children


def format_tree() -> str:
    children = children_map()
    lines: list[str] = []

    def walk(index: int, prefix: str, is_last: bool) -> None:
        branch = "└─ " if is_last else "├─ "
        name = JOINT_NAMES[index]
        pose_slice = f"pose[{index * 3:02d}:{index * 3 + 3:02d}]"
        lines.append(
            f"{prefix}{branch}{index:02d} {name:<8} {ANATOMICAL[name]:<12} {pose_slice}"
        )
        kids = children[index]
        next_prefix = prefix + ("   " if is_last else "│  ")
        for kid_i, kid in enumerate(kids):
            walk(kid, next_prefix, kid_i == len(kids) - 1)

    lines.append("00 wrist    wrist        pose[00:03]  <- global rotation, not inside the 45-D finger pose")
    kids = children[0]
    for kid_i, kid in enumerate(kids):
        walk(kid, "", kid_i == len(kids) - 1)
    return "\n".join(lines)


def bone_segments(positions: np.ndarray) -> list[tuple[int, int]]:
    return [(int(parent), index) for index, parent in enumerate(PARENTS) if parent >= 0]
