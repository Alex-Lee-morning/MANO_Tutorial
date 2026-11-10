"""Lab for TUTORIAL.md. Read a chapter, then run the matching number.

    python tutorial.py       # every figure
    python tutorial.py 6     # one chapter
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mano_skeleton as sk
from official_mano import require_model

OUT = Path(__file__).resolve().parent / "out" / "tutorial"
PALM = np.array([0.0, -1.0, 0.0])


def _header(number: int, title: str) -> None:
    print()
    print("=" * 72)
    print(f"第 {number} 章  {title}")
    print("=" * 72)


def _mesh(path: Path, panels: list[tuple[str, dict]], faces: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for font_path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    cloud = np.vstack([item["vertices"] for _, item in panels])
    columns = len(panels)
    fig = plt.figure(figsize=(4.4 * columns, 4.6))
    for index, (title, result) in enumerate(panels, start=1):
        axis = fig.add_subplot(1, columns, index, projection="3d")
        triangles = result["vertices"][faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        facing = normals @ PALM > 0.0
        shade = 0.55 + 0.45 * np.abs(normals[facing] @ PALM)
        color = np.zeros((int(facing.sum()), 4))
        color[:, 0] = 0.86 * shade
        color[:, 1] = 0.64 * shade
        color[:, 2] = 0.50 * shade
        color[:, 3] = 1.0
        axis.add_collection3d(Poly3DCollection(triangles[facing], facecolors=color, linewidths=0))
        tips = np.stack([result["tips"][name] for name in ("thumb", "index", "middle", "ring", "pinky")])
        axis.scatter(tips[:, 0], tips[:, 1], tips[:, 2], s=18, c="#b91c1c", depthshade=False)
        axis.set_title(title, fontsize=11)
        axis.view_init(elev=0, azim=90)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        center = cloud.mean(axis=0)
        radius = 0.55 * np.max(np.linalg.norm(cloud - center, axis=1))
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"图：{path}")


def chapter_2() -> None:
    _header(2, "16 个关节")
    print(sk.format_tree())
    print()
    print("hand_pose 只有 45 个数，从食指 MCP 开始，不含手腕。")
    print("关节 i>=1 在 hand_pose 里的位置是 [3*(i-1) : 3*(i-1)+3]。")


def chapter_4(model) -> None:
    _header(4, "伸直模板和训练平均姿态")
    flat = model.forward(flat_hand=True)
    mean = model.forward(flat_hand=False)
    direction = flat["joints"][4] - flat["joints"][0]
    print(f"顶点 {flat['vertices'].shape}，关节 {flat['joints'].shape}")
    print("腕 -> 中指 MCP（米）:", np.round(direction, 4))
    print("食指尖 伸直:", np.round(flat["tips"]["index"], 4))
    print("食指尖 平均:", np.round(mean["tips"]["index"], 4))
    print("平均姿态把食指尖往 -Y 拉了 {:.1f} mm。掌心在 -Y。".format(
        (flat["tips"]["index"][1] - mean["tips"]["index"][1]) * 1000
    ))
    _mesh(
        OUT / "04_flat_and_mean.png",
        [("伸直模板  betas=0, 轴角=0", flat), ("训练平均姿态  加上 hands_mean", mean)],
        model.faces,
    )


def chapter_6(model) -> None:
    _header(6, "只弯曲食指")
    flat = model.forward(flat_hand=True)
    pose = np.zeros(45)
    # 食指 MCP / PIP / DIP 的局部 Z。正角度把指尖推向 -Y，也就是掌心。
    pose[2], pose[5], pose[8] = np.deg2rad([40.0, 65.0, 45.0])
    curled = model.forward(hand_pose=pose, flat_hand=True)
    twist = np.zeros(45)
    twist[0], twist[3], twist[6] = np.deg2rad([40.0, 65.0, 45.0])
    twisted = model.forward(hand_pose=twist, flat_hand=True)
    curl_move = np.linalg.norm(curled["tips"]["index"] - flat["tips"]["index"])
    twist_move = np.linalg.norm(twisted["tips"]["index"] - flat["tips"]["index"])
    others = [name for name in ("middle", "ring", "pinky", "thumb")]
    leak = max(np.linalg.norm(curled["tips"][name] - flat["tips"][name]) for name in others)
    print(f"绕 Z 弯曲，食指尖移动 {curl_move * 1000:.1f} mm")
    print(f"同样的角度绕 X（沿手指扭转），食指尖只移动 {twist_move * 1000:.1f} mm")
    print(f"其他四指指尖的最大移动 {leak * 1000:.2f} mm")
    if curl_move < 0.02 or twist_move > curl_move or leak > 0.005:
        raise RuntimeError("食指屈曲实验和官方坐标系不一致")
    _mesh(
        OUT / "06_index_curl.png",
        [("伸直", flat), ("食指绕局部 Z 弯曲", curled)],
        model.faces,
    )


def chapter_7(model) -> None:
    _header(7, "形状只改变这是谁的手")
    panels = []
    lengths = []
    for beta0 in (-2.0, 0.0, 2.0):
        betas = np.zeros(10)
        betas[0] = beta0
        result = model.forward(betas=betas, flat_hand=True)
        length = np.linalg.norm(result["tips"]["middle"] - result["joints"][0])
        lengths.append(length)
        panels.append((f"beta[0] = {beta0:+.0f}", result))
        print(f"beta[0]={beta0:+.0f}  腕到中指尖 {length * 1000:.1f} mm")
    print("三只手的姿态都是伸直。变的是长短，不是握拳。")
    _mesh(OUT / "07_shape.png", panels, model.faces)


def chapter_8() -> None:
    _header(8, "公式和文件里的数组")
    print(
        """
v_shaped = v_template + shapedirs · betas
J        = J_regressor · v_shaped
v_posed  = v_shaped + posedirs · (R - I)     # 只有 15 个手指关节
vertices = LBS(v_posed, J, pose, weights)
joints   = FK(J, pose)

玩具手指把同一套式子做在 5 个顶点上：python course.py 5
然后 python course.py 6 和 python course.py 7。
"""
    )


def chapter_9(model) -> None:
    _header(9, "两个零")
    flat = model.forward(hand_pose=np.zeros(45), flat_hand=True)
    mean = model.forward(hand_pose=np.zeros(45), flat_hand=False)
    gap = np.linalg.norm(mean["joints"] - flat["joints"])
    print(f"flat_hand=True  且 hand_pose=0：伸直模板")
    print(f"flat_hand=False 且 hand_pose=0：训练平均姿态，关节差 {gap * 1000:.1f} mm")
    print("手腕坐标两边相同，因为 hands_mean 不含全局旋转。")
    print("腕 伸直", np.round(flat["joints"][0], 4))
    print("腕 平均", np.round(mean["joints"][0], 4))


def chapter_10(model) -> None:
    _header(10, "指尖是顶点")
    flat = model.forward(flat_hand=True)
    print(f"{'名字':<8} {'顶点号':>6} {'坐标 (米)'}")
    for name, vertex_id in sk.TIP_VERTEX_IDS.items():
        point = flat["tips"][name]
        same = np.allclose(point, flat["vertices"][vertex_id])
        print(f"{name:<8} {vertex_id:6d}  {point[0]:7.4f} {point[1]:7.4f} {point[2]:7.4f}  与网格一致={same}")
    print("关节 15 是拇指 IP，不是拇指尖。拇指尖是顶点 744。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MANO tutorial labs")
    parser.add_argument("chapter", nargs="?", type=int)
    args = parser.parse_args()
    chosen = args.chapter
    needs_model = chosen in (None, 4, 6, 7, 9, 10)
    model = require_model() if needs_model else None
    if chosen in (None, 2):
        chapter_2()
    if chosen in (None, 4):
        chapter_4(model)
    if chosen in (None, 6):
        chapter_6(model)
    if chosen in (None, 7):
        chapter_7(model)
    if chosen in (None, 8):
        chapter_8()
    if chosen in (None, 9):
        chapter_9(model)
    if chosen in (None, 10):
        chapter_10(model)
    if chosen not in (None, 2, 4, 6, 7, 8, 9, 10):
        raise SystemExit("章节号用 2、4、6、7、8、9、10。阅读部分在 TUTORIAL.md。")
    print()
    print("课文：TUTORIAL.md")


if __name__ == "__main__":
    main()
