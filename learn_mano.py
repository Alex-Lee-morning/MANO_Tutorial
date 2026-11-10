"""Three views of the MANO right hand: tree, schematic demo, official model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mano_skeleton as sk
from official_mano import ManoModelMissing, require_model

OUT = Path(__file__).resolve().parent / "out"


def command_tree() -> None:
    print("MANO right hand: 16 joints. The pose is 48 numbers = global rotation 3 + fingers 45.")
    print("The 45 finger numbers follow the tree below, 3 axis-angle values per joint, in radians.")
    print("Pinky is stored before ring.")
    print()
    print(sk.format_tree())
    print()
    print("Fingertips are not in this tree. They are mesh vertices. Common ids:")
    for name, vertex_id in sk.TIP_VERTEX_IDS.items():
        print(f"  {name:<6} vertex {vertex_id}")
    print()
    print("On this teaching skeleton, a positive local X rotation flexes a finger toward the palm (+Z).")
    print("All-zero axis-angles are the flat template. Adding the official hands_mean bends the fingers slightly.")


def _demo_poses() -> dict[str, np.ndarray]:
    flat = sk.zero_pose()
    curl = sk.pose_from_dict(
        {
            "index1": (55, 0, 0),
            "index2": (70, 0, 0),
            "index3": (45, 0, 0),
            "middle1": (50, 0, 0),
            "middle2": (65, 0, 0),
            "middle3": (40, 0, 0),
            "ring1": (50, 0, 0),
            "ring2": (65, 0, 0),
            "ring3": (40, 0, 0),
            "pinky1": (45, 0, 0),
            "pinky2": (60, 0, 0),
            "pinky3": (35, 0, 0),
            "thumb1": (0, 25, 35),
            "thumb2": (20, 0, 15),
            "thumb3": (15, 0, 10),
        }
    )
    return {"flat": flat, "curl": curl}


def command_demo() -> None:
    poses = _demo_poses()
    flat_pos, flat_ori = sk.forward_kinematics(poses["flat"])
    curl_pos, curl_ori = sk.forward_kinematics(poses["curl"])
    flat_tips = sk.fingertip_positions(flat_pos, flat_ori)
    curl_tips = sk.fingertip_positions(curl_pos, curl_ori)

    print("Teaching skeleton only. These millimetres are not the official MANO template.")
    print("Index fingertip Z: flat {:.1f} mm -> curled {:.1f} mm (larger means toward the palm).".format(
        flat_tips["index"][2], curl_tips["index"][2]
    ))
    if curl_tips["index"][2] <= flat_tips["index"][2]:
        raise RuntimeError("Index curl did not move the tip toward +Z")

    print()
    print(f"{'joint':<10} {'flat XYZ (mm)':<28} {'curled XYZ (mm)'}")
    for index, name in enumerate(sk.JOINT_NAMES):
        a = flat_pos[index]
        b = curl_pos[index]
        print(
            f"{index:02d} {name:<8} "
            f"{a[0]:7.1f} {a[1]:7.1f} {a[2]:7.1f}    "
            f"{b[0]:7.1f} {b[1]:7.1f} {b[2]:7.1f}"
        )

    out = OUT / "schematic_hand.png"
    _draw(poses, out)
    print()
    print(f"Wrote {out}")


def command_official() -> None:
    try:
        model = require_model()
    except ManoModelMissing as exc:
        print(exc)
        return

    flat = model.forward(flat_hand=True)
    mean_pose = model.forward(flat_hand=False)
    betas = np.zeros(10)
    betas[0] = 1.0
    shaped = model.forward(betas=betas, flat_hand=True)

    print(f"Loaded {model.path}")
    print(f"vertices {flat['vertices'].shape[0]}, faces {model.faces.shape[0]}, joints {flat['joints'].shape[0]}")
    print(f"shape basis {model.shapedirs.shape[-1]}, pose-blend last dim {model.posedirs.shape[-1]}")
    print()
    print(f"{'joint':<10} {'flat template':^32} {'plus hands_mean':^32} {'beta0=+1'}")
    for index, name in enumerate(sk.JOINT_NAMES):
        a, b, c = flat["joints"][index], mean_pose["joints"][index], shaped["joints"][index]
        print(
            f"{index:02d} {name:<8} "
            f"{a[0]:7.4f} {a[1]:7.4f} {a[2]:7.4f}   "
            f"{b[0]:7.4f} {b[1]:7.4f} {b[2]:7.4f}   "
            f"{c[0]:7.4f} {c[1]:7.4f} {c[2]:7.4f}"
        )
    print()
    print("Fingertip vertices (metres):")
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        point = flat["tips"][name]
        print(f"  {name:<6} {point[0]:7.4f} {point[1]:7.4f} {point[2]:7.4f}")

    out = OUT / "official_joints.npy"
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(out, flat["joints"])
    print()
    print(f"Saved the 16 flat-template joints to {out}, metres, shape (16, 3).")
    _draw_official(flat["joints"], flat["tips"], OUT / "official_hand.png")
    mesh_path = OUT / "official_right_hand.png"
    _draw_official_mesh(model.faces, flat, mean_pose, mesh_path)
    print(f"Saved the right-hand mesh to {mesh_path}.")


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _draw(poses: dict[str, np.ndarray], out: Path) -> None:
    plt = _setup_matplotlib()
    fig = plt.figure(figsize=(12, 5.2))
    axes = [
        fig.add_subplot(1, 3, 1),
        fig.add_subplot(1, 3, 2),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    titles = ["Palm  X thumb->pinky, Y wrist->tips", "Side  Y wrist->tips, Z toward palm", "3D"]
    for title, axis in zip(titles, axes):
        axis.set_title(title, fontsize=11)

    for label, pose in poses.items():
        positions, orientations = sk.forward_kinematics(pose)
        tips = sk.fingertip_positions(positions, orientations)
        linestyle = "-" if label == "flat" else "--"
        _plot_2d(axes[0], positions, tips, 0, 1, linestyle, label)
        _plot_2d(axes[1], positions, tips, 1, 2, linestyle, label)
        _plot_3d(axes[2], positions, tips, linestyle, label)

    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[0].legend(frameon=False)
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (mm)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_2d(axis, positions, tips, a, b, linestyle, label) -> None:
    for parent, child in sk.bone_segments(positions):
        color = sk.FINGER_COLOR[sk.finger_of(child)]
        axis.plot(
            [positions[parent, a], positions[child, a]],
            [positions[parent, b], positions[child, b]],
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    axis.scatter(positions[:, a], positions[:, b], s=18, c="#111827", zorder=3)
    tip_xy = np.stack([tips[name] for name in ("thumb", "index", "middle", "ring", "pinky")])
    axis.scatter(tip_xy[:, a], tip_xy[:, b], s=28, marker="x", c="#111827", label=label)


def _plot_3d(axis, positions, tips, linestyle, label) -> None:
    for parent, child in sk.bone_segments(positions):
        color = sk.FINGER_COLOR[sk.finger_of(child)]
        axis.plot(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 1], positions[child, 1]],
            [positions[parent, 2], positions[child, 2]],
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=12, c="#111827")
    tip_xyz = np.stack(list(tips.values()))
    axis.scatter(tip_xyz[:, 0], tip_xyz[:, 1], tip_xyz[:, 2], s=20, marker="x", c="#111827", label=label)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    _equal_3d(axis, np.vstack([positions, tip_xyz]))


def _draw_official(joints: np.ndarray, tips: dict[str, np.ndarray], out: Path) -> None:
    plt = _setup_matplotlib()
    fig = plt.figure(figsize=(6, 6))
    axis = fig.add_subplot(111, projection="3d")
    for parent, child in sk.bone_segments(joints):
        color = sk.FINGER_COLOR[sk.finger_of(child)]
        axis.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=color,
            linewidth=2,
        )
    axis.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=16, c="#111827")
    for index, name in enumerate(sk.JOINT_NAMES):
        axis.text(joints[index, 0], joints[index, 1], joints[index, 2], f" {index}", fontsize=8)
    tip_xyz = np.stack([tips[name] for name in ("thumb", "index", "middle", "ring", "pinky")])
    axis.scatter(tip_xyz[:, 0], tip_xyz[:, 1], tip_xyz[:, 2], marker="x", c="#b91c1c")
    axis.set_title("Official MANO flat template, 16 joints (metres)")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    _equal_3d(axis, np.vstack([joints, tip_xyz]))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _draw_official_mesh(faces: np.ndarray, flat: dict, mean_pose: dict, out: Path) -> None:
    """Two poses, each from both sides of the hand. Fingers point along -X."""
    plt = _setup_matplotlib()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 8))
    panels = (
        (flat, "Flat template, from +Y", np.array([0.0, 1.0, 0.0]), 0, -90),
        (flat, "Flat template, from -Y", np.array([0.0, -1.0, 0.0]), 0, 90),
        (mean_pose, "Training-mean pose, from +Y", np.array([0.0, 1.0, 0.0]), 0, -90),
        (mean_pose, "Training-mean pose, from -Y", np.array([0.0, -1.0, 0.0]), 0, 90),
    )
    cloud = np.vstack([flat["vertices"], mean_pose["vertices"]])
    for index, (result, title, view, elev, azim) in enumerate(panels, start=1):
        axis = fig.add_subplot(2, 2, index, projection="3d")
        triangles = result["vertices"][faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(length, 1e-12)
        facing = normals @ view > 0.0
        shade = 0.55 + 0.45 * np.abs(normals[facing] @ view)
        color = np.zeros((int(facing.sum()), 4))
        color[:, 0] = 0.86 * shade
        color[:, 1] = 0.64 * shade
        color[:, 2] = 0.50 * shade
        color[:, 3] = 1.0
        axis.add_collection3d(Poly3DCollection(triangles[facing], facecolors=color, linewidths=0))
        joints = result["joints"]
        for parent, child in sk.bone_segments(joints):
            axis.plot(
                [joints[parent, 0], joints[child, 0]],
                [joints[parent, 1], joints[child, 1]],
                [joints[parent, 2], joints[child, 2]],
                color=sk.FINGER_COLOR[sk.finger_of(child)],
                linewidth=1.2,
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.view_init(elev=elev, azim=azim)
        _equal_3d(axis, cloud)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _equal_3d(axis, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = 0.55 * np.max(np.linalg.norm(points - center, axis=1))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the MANO right-hand skeleton")
    parser.add_argument("command", nargs="?", choices=("tree", "demo", "official", "all"), default="all")
    args = parser.parse_args()
    if args.command in ("tree", "all"):
        command_tree()
        print()
    if args.command in ("demo", "all"):
        command_demo()
        print()
    if args.command == "official":
        command_official()


if __name__ == "__main__":
    main()
