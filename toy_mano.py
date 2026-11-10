"""A 4-joint finger that uses the same equations as MANO.

Nothing in this file is an official MANO weight. The finger exists so the
shape blend, pose blend, joint regressor, and linear blend skinning can be
read one vertex at a time. The real model is the same arithmetic on 778
vertices and 16 joints.
"""

from __future__ import annotations

import numpy as np

from mano_skeleton import rodrigues

NAMES = ("wrist", "mcp", "pip", "dip")
PARENTS = np.array([-1, 0, 1, 2], dtype=np.int64)

# Vertices 0-3 sit on the four joints. Vertex 4 is skin beside the PIP,
# offset in Z so a PIP rotation moves it. Units are arbitrary, not metres.
V_TEMPLATE = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 2.0, 0.3],
    ],
    dtype=np.float64,
)

# One shape component: lengthen the finger past the MCP.
SHAPEDIRS = np.zeros((5, 3, 1), dtype=np.float64)
SHAPEDIRS[2, 1, 0] = 0.2
SHAPEDIRS[3, 1, 0] = 0.4

# Joint i is vertex i. Vertex 4 is skin and does not define a joint.
# Real MANO uses a sparse regressor, and its weights may be negative.
J_REGRESSOR = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Vertex 4 is shared by MCP and PIP. The joint vertices are rigid.
WEIGHTS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.25, 0.75, 0.0],
    ],
    dtype=np.float64,
)

# 3 non-root joints x 9 entries of (R - I) = 27.
# PIP is non-root slot 1, so its nine numbers start at index 9.
# Row-major entry (2, 1) of (R - I) is local index 7, absolute channel 16.
# For a rotation about X that entry equals sin(theta), so the skin vertex
# beside the PIP receives (0, 0, 0.05 * sin(theta)).
POSEDIRS = np.zeros((5, 3, 27), dtype=np.float64)
POSEDIRS[4, 2, 16] = 0.05


def pose_feature(full_pose: np.ndarray) -> np.ndarray:
    """(R - I) for every non-root joint. The root rotation is not included."""
    pose = np.asarray(full_pose, dtype=np.float64).reshape(-1)
    parts = []
    for index in range(1, 4):
        rotation = rodrigues(pose[index * 3 : index * 3 + 3])
        parts.append((rotation - np.eye(3)).reshape(-1))
    return np.concatenate(parts)


def forward(
    full_pose: np.ndarray | None = None,
    betas: np.ndarray | None = None,
    *,
    use_pose_blend: bool = True,
) -> dict[str, np.ndarray]:
    """MANO forward pass, reduced to one finger.

    full_pose: (4, 3) or (12,) axis-angle. Entry 0 is the global rotation.
    betas: (1,) shape coefficient. None means the template shape.
    """
    pose = np.zeros(12) if full_pose is None else np.asarray(full_pose, dtype=np.float64).reshape(-1)
    if pose.shape != (12,):
        raise ValueError(f"full_pose must have 12 numbers, got {pose.shape}")
    beta = np.zeros(1) if betas is None else np.asarray(betas, dtype=np.float64).reshape(-1)

    v_shaped = V_TEMPLATE + np.einsum("vck,k->vc", SHAPEDIRS, beta)
    joints_rest = J_REGRESSOR @ v_shaped
    feature = pose_feature(pose)
    correction = np.einsum("vcp,p->vc", POSEDIRS, feature) if use_pose_blend else 0.0
    v_posed = v_shaped + correction
    joints = _fk(pose, joints_rest)
    vertices = _lbs(pose, joints_rest, v_posed)
    return {
        "vertices": vertices,
        "joints": joints,
        "joints_rest": joints_rest,
        "v_shaped": v_shaped,
        "v_posed": v_posed,
        "pose_feature": feature,
    }


def _fk(full_pose: np.ndarray, joints_rest: np.ndarray) -> np.ndarray:
    positions = np.zeros((4, 3), dtype=np.float64)
    orientations = np.zeros((4, 3, 3), dtype=np.float64)
    for index in range(4):
        local = rodrigues(full_pose[index * 3 : index * 3 + 3])
        parent = int(PARENTS[index])
        if parent < 0:
            orientations[index] = local
            positions[index] = joints_rest[index]
        else:
            orientations[index] = orientations[parent] @ local
            offset = joints_rest[index] - joints_rest[parent]
            positions[index] = positions[parent] + orientations[parent] @ offset
    return positions


def _lbs(full_pose: np.ndarray, joints_rest: np.ndarray, v_posed: np.ndarray) -> np.ndarray:
    positions = _fk(full_pose, joints_rest)
    orientations = np.zeros((4, 3, 3), dtype=np.float64)
    for index in range(4):
        local = rodrigues(full_pose[index * 3 : index * 3 + 3])
        parent = int(PARENTS[index])
        orientations[index] = local if parent < 0 else orientations[parent] @ local

    transforms = np.zeros((4, 4, 4), dtype=np.float64)
    for index in range(4):
        transforms[index, :3, :3] = orientations[index]
        transforms[index, :3, 3] = positions[index] - orientations[index] @ joints_rest[index]
        transforms[index, 3, 3] = 1.0

    weighted = np.einsum("vj,jab->vab", WEIGHTS, transforms)
    posed_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
    return np.einsum("vab,vb->va", weighted, posed_h)[:, :3]
