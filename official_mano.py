"""Load an official MANO pickle and run shape blend, pose blend, and linear blend skinning.

Download MANO_RIGHT.pkl yourself from https://mano.is.tue.mpg.de/.
This module does not contain or download MANO weights.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from mano_skeleton import JOINT_NAMES, PARENTS, TIP_VERTEX_IDS, rodrigues

ROOT = Path(__file__).resolve().parent
_RELEASE = ROOT / "mano_v1_2" / "models"
DEFAULT_RIGHT = _RELEASE / "MANO_RIGHT.pkl"
DEFAULT_LEFT = _RELEASE / "MANO_LEFT.pkl"


class ManoModelMissing(FileNotFoundError):
    pass


def _to_numpy(value) -> np.ndarray:
    """Turn a pickle value into an array.

    MANO_RIGHT.pkl stores shapedirs as a chumpy Select over 20 components,
    keeping the first 10. chumpy itself is not required.
    """
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    indices = getattr(value, "idxs", None)
    source = getattr(value, "a", None)
    if indices is not None and source is not None:
        flat = _to_numpy(getattr(source, "x", source)).ravel()
        selected = flat[np.asarray(indices)]
        shape = getattr(value, "preferred_shape", None)
        return selected.reshape(shape) if shape is not None else selected
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict) and "x" in state and not callable(state["x"]):
        return _to_numpy(state["x"])
    if hasattr(value, "r") and not callable(value.r):
        return np.asarray(value.r)
    return np.asarray(value)


class _TolerantUnpickler(pickle.Unpickler):
    """The official pickle was saved with chumpy. Recover arrays if chumpy is absent."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            return type(name, (), {"__module__": module, "__setstate__": _setstate})


def _setstate(self, state):
    if isinstance(state, dict):
        self.__dict__.update(state)
    else:
        self._state = state


def load_mano_pkl(path: Path) -> dict:
    with path.open("rb") as handle:
        try:
            raw = pickle.load(handle, encoding="latin1")
        except ModuleNotFoundError:
            handle.seek(0)
            raw = _TolerantUnpickler(handle, encoding="latin1").load()
    if not isinstance(raw, dict):
        raise TypeError(f"{path} is not a MANO dictionary pickle")
    return raw


class OfficialMano:
    def __init__(self, path: Path):
        raw = load_mano_pkl(path)
        self.path = path
        self.v_template = _to_numpy(raw["v_template"]).astype(np.float64)
        self.shapedirs = _to_numpy(raw["shapedirs"]).astype(np.float64)
        self.posedirs = _to_numpy(raw["posedirs"]).astype(np.float64)
        self.weights = _to_numpy(raw["weights"]).astype(np.float64)
        self.j_regressor = _to_numpy(raw["J_regressor"]).astype(np.float64)
        self.faces = _to_numpy(raw["f"]).astype(np.int64)
        self.hands_mean = _to_numpy(raw["hands_mean"]).astype(np.float64).reshape(-1)
        self.hands_components = _to_numpy(raw["hands_components"]).astype(np.float64)

        kintree = _to_numpy(raw["kintree_table"]).astype(np.int64)
        parents = kintree[0].copy()
        parents[parents < 0] = -1
        parents[parents > 1000] = -1
        if parents.shape != PARENTS.shape or np.any(parents != PARENTS):
            raise ValueError(
                f"kintree parents do not match the course notes: {parents.tolist()}"
            )
        self.parents = parents

        if self.v_template.shape != (778, 3):
            raise ValueError(f"v_template should be (778, 3), got {self.v_template.shape}")
        if self.j_regressor.shape[0] != 16:
            raise ValueError(f"J_regressor should have 16 rows, got {self.j_regressor.shape}")

    def forward(
        self,
        global_orient: np.ndarray | None = None,
        hand_pose: np.ndarray | None = None,
        betas: np.ndarray | None = None,
        transl: np.ndarray | None = None,
        *,
        flat_hand: bool = True,
        pose_is_pca: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return vertices, 16 joints, and 5 fingertip vertices.

        flat_hand=True: hand_pose is the 15 joint axis-angles. Zeros are the flat template.
        flat_hand=False: zeros are added to hands_mean, so the fingers sit at the training average.
        pose_is_pca=True: hand_pose holds PCA coefficients and is multiplied by hands_components.
        """
        global_orient = np.zeros(3) if global_orient is None else np.asarray(global_orient, dtype=np.float64).reshape(3)
        betas = np.zeros(self.shapedirs.shape[-1]) if betas is None else np.asarray(betas, dtype=np.float64).reshape(-1)
        transl = np.zeros(3) if transl is None else np.asarray(transl, dtype=np.float64).reshape(3)

        if hand_pose is None:
            finger_pose = np.zeros(45, dtype=np.float64)
        else:
            finger_pose = np.asarray(hand_pose, dtype=np.float64).reshape(-1)
            if pose_is_pca:
                comps = np.asarray(self.hands_components, dtype=np.float64)
                if comps.shape[0] != finger_pose.shape[0] and comps.shape[1] == finger_pose.shape[0]:
                    comps = comps.T
                comps = comps[: finger_pose.shape[0]]
                finger_pose = finger_pose @ comps
            if finger_pose.shape != (45,):
                raise ValueError(f"finger pose should be 45 axis-angle values, got {finger_pose.shape}")

        if not flat_hand:
            finger_pose = finger_pose + self.hands_mean
        full_pose = np.concatenate([global_orient, finger_pose])

        n_betas = min(betas.shape[0], self.shapedirs.shape[-1])
        v_shaped = self.v_template + np.einsum(
            "vck,k->vc", self.shapedirs[..., :n_betas], betas[:n_betas]
        )
        joints_rest = self.j_regressor @ v_shaped

        pose_feature = _pose_feature(full_pose)
        posedirs = _as_posedirs(self.posedirs)
        v_posed = v_shaped + np.einsum("vcp,p->vc", posedirs, pose_feature)

        transforms = _skinning_transforms(full_pose, joints_rest, self.parents)
        weighted = np.einsum("vj,jab->vab", self.weights, transforms)
        posed_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
        vertices = np.einsum("vab,vb->va", weighted, posed_h)[:, :3] + transl
        joints = _fk_joints(full_pose, joints_rest, self.parents) + transl

        tips = {name: vertices[vertex_id] for name, vertex_id in TIP_VERTEX_IDS.items()}
        return {
            "vertices": vertices,
            "joints": joints,
            "tips": tips,
            "full_pose": full_pose,
            "joints_rest": joints_rest,
        }


def _as_posedirs(posedirs: np.ndarray) -> np.ndarray:
    """Return posedirs as (778, 3, 135): nine entries of (R - I) for each of 15 joints."""
    array = np.asarray(posedirs, dtype=np.float64)
    if array.shape == (778, 3, 135):
        return array
    if array.shape == (135, 778, 3):
        return np.moveaxis(array, 0, -1)
    if array.shape == (778 * 3, 135):
        return array.reshape(778, 3, 135)
    raise ValueError(f"unrecognized posedirs shape {array.shape}")


def _pose_feature(full_pose: np.ndarray) -> np.ndarray:
    """Stack (R - I) for the 15 non-root joints into 135 numbers. Global rotation is excluded."""
    parts = []
    for index in range(1, 16):
        rotation = rodrigues(full_pose[index * 3 : index * 3 + 3])
        parts.append((rotation - np.eye(3)).reshape(-1))
    return np.concatenate(parts)


def _fk_joints(full_pose: np.ndarray, joints_rest: np.ndarray, parents: np.ndarray) -> np.ndarray:
    positions = np.zeros((16, 3), dtype=np.float64)
    orientations = np.zeros((16, 3, 3), dtype=np.float64)
    for index in range(16):
        local = rodrigues(full_pose[index * 3 : index * 3 + 3])
        parent = int(parents[index])
        if parent < 0:
            orientations[index] = local
            positions[index] = joints_rest[index]
        else:
            orientations[index] = orientations[parent] @ local
            offset = joints_rest[index] - joints_rest[parent]
            positions[index] = positions[parent] + orientations[parent] @ offset
    return positions


def _skinning_transforms(full_pose: np.ndarray, joints_rest: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Per-joint 4x4 transforms that take a rest-pose point into the posed hand."""
    positions = _fk_joints(full_pose, joints_rest, parents)
    orientations = np.zeros((16, 3, 3), dtype=np.float64)
    for index in range(16):
        local = rodrigues(full_pose[index * 3 : index * 3 + 3])
        parent = int(parents[index])
        orientations[index] = local if parent < 0 else orientations[parent] @ local

    transforms = np.zeros((16, 4, 4), dtype=np.float64)
    for index in range(16):
        transforms[index, :3, :3] = orientations[index]
        transforms[index, :3, 3] = positions[index] - orientations[index] @ joints_rest[index]
        transforms[index, 3, 3] = 1.0
    return transforms


def require_model(path: Path | None = None) -> OfficialMano:
    model_path = DEFAULT_RIGHT if path is None else Path(path)
    if not model_path.is_file():
        raise ManoModelMissing(
            "Official model file is missing.\n"
            f"Download MANO_RIGHT.pkl and place it at:\n  {DEFAULT_RIGHT}\n"
            "https://mano.is.tue.mpg.de/\n"
            "Until then, run: python course.py"
        )
    return OfficialMano(model_path)
