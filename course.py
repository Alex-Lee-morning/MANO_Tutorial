"""Runnable half of the MANO course. Read COURSE.md, then run a lesson.

    python course.py          # lessons 0 through 10
    python course.py 7        # one lesson
    python course.py 11       # official pickle, after you download it
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize_scalar

import mano_skeleton as sk
import toy_mano as toy
from official_mano import ManoModelMissing, require_model

LESSON_TITLES = {
    0: "What MANO computes",
    1: "The 16 joints and the 48 pose numbers",
    2: "Axis-angle",
    3: "Forward kinematics",
    4: "What is inside the pickle",
    5: "Shape blend shapes",
    6: "Pose blend shapes",
    7: "Linear blend skinning",
    8: "The two zeros, and PCA",
    9: "Fingertips and the left hand",
    10: "Fitting is an optimization",
    11: "The official model",
}


def _header(number: int) -> None:
    print()
    print("=" * 72)
    print(f"Lesson {number}. {LESSON_TITLES[number]}")
    print("=" * 72)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  check passed: {message}")


def lesson_0() -> None:
    _header(0)
    print(
        """
MANO is a function from body-shape and pose numbers to a hand mesh.

    vertices = LBS( template + shape(betas) + pose_blend(pose), joints(betas), pose )

The released right hand has:
    778 vertices, 1538 triangles, 16 joints.

The skeleton is not a separate asset. Joints are a linear function of the
vertices after the shape blend. Pose then moves those joints with a kinematic
tree, and the mesh follows by linear blend skinning plus a corrective
displacement called a pose blend shape.

Lessons 1-4 name every index and every array.
Lessons 5-7 run the formula on a 5-vertex finger, which is the same arithmetic.
Lessons 8-10 cover the two kinds of zero pose, the left hand, and fitting.
Lesson 11 runs the formula on MANO_RIGHT.pkl once you have downloaded it.
"""
    )


def lesson_1() -> None:
    _header(1)
    print(sk.format_tree())
    print()
    print("Joint i occupies pose[3*i : 3*i+3]. Joint 0 is the wrist, and it is global.")
    _check(sk.JOINT_NAMES[7] == "pinky1", "index 7 is pinky1, not ring1")
    _check(sk.JOINT_NAMES[10] == "ring1", "index 10 is ring1")
    _check(int(sk.PARENTS[10]) == 0, "ring MCP is a child of the wrist, not of the pinky")
    _check(sk.ANATOMICAL["thumb1"] == "thumb CMC", "thumb1 is the CMC joint")
    _check(len(sk.JOINT_NAMES) == 16, "there is no 17th kinematic joint")
    print()
    print("A 45-D finger pose drops joint 0. A 48-D pose keeps it at the front.")
    print("Fingertip vertex ids (not joints):", sk.TIP_VERTEX_IDS)


def lesson_2() -> None:
    _header(2)
    axis = np.array([np.pi / 2, 0.0, 0.0])
    rotation = sk.rodrigues(axis)
    y_axis = np.array([0.0, 1.0, 0.0])
    image = rotation @ y_axis
    print("90 degrees about +X sends +Y to:")
    print(" ", np.round(image, 5))
    print("Right-hand rule: +Y rotates toward +Z.")
    _check(np.allclose(image, [0.0, 0.0, 1.0], atol=1e-8), "+Y goes to +Z")
    _check(np.allclose(sk.rodrigues(np.zeros(3)), np.eye(3)), "zero axis-angle is the identity")
    print()
    print("MANO stores radians, not degrees. At rest, joint frames match the model axes.")
    print("They are not aligned with the bone. A 'flexion' is whichever axis is")
    print("perpendicular to that finger in the template, not automatically local X")
    print("on the official mesh. The teaching skeleton does use local +X for flexion.")


def lesson_3() -> None:
    _header(3)
    rest, _ = sk.forward_kinematics(sk.zero_pose())
    bent = sk.zero_pose()
    bent[2] = np.array([np.deg2rad(40.0), 0.0, 0.0])  # index2, the PIP
    moved, _ = sk.forward_kinematics(bent)
    print("Rotate only index2 (PIP) by 40 degrees about X.")
    print(f"  index2 moved? {np.linalg.norm(moved[2] - rest[2]):.3e} mm")
    print(f"  index3 moved? {np.linalg.norm(moved[3] - rest[3]):.3f} mm")
    print(f"  wrist  moved? {np.linalg.norm(moved[0] - rest[0]):.3e} mm")
    _check(np.allclose(moved[2], rest[2]), "a joint's own rotation does not translate it")
    _check(np.linalg.norm(moved[3] - rest[3]) > 1.0, "the child joint does move")
    _check(np.allclose(moved[0], rest[0]), "a distal rotation does not move the wrist")
    print()
    print("Bone lengths in learn_mano.py are a drawing. Do not quote them as MANO.")


def lesson_4() -> None:
    _header(4)
    rows = [
        ("v_template", "(778, 3)", "mean hand, metres"),
        ("f", "(1538, 3)", "triangle vertex indices"),
        ("shapedirs", "(778, 3, 10)", "usually 10 shape components"),
        ("posedirs", "(778, 3, 135)", "15 joints x 9 entries of (R-I)"),
        ("J_regressor", "(16, 778)", "joints from shaped vertices; weights may be negative"),
        ("weights", "(778, 16)", "LBS weights; each row sums to 1"),
        ("kintree_table", "(2, 16)", "row 0 is the parent id"),
        ("hands_components", "(45, 45)", "PCA basis of the 45-D finger pose"),
        ("hands_mean", "(45,)", "mean finger pose of the training registrations"),
    ]
    for name, shape, meaning in rows:
        print(f"  {name:<18} {shape:<16} {meaning}")
    print()
    print("The root parent is often stored as the unsigned integer 4294967295, which is -1.")
    print("The file is a chumpy pickle. You download it; it is not in this folder.")
    print("The release copy is mano_v1_2/models/MANO_RIGHT.pkl. Then run lesson 11.")


def lesson_5() -> None:
    _header(5)
    mean = toy.forward(betas=[0.0])
    long = toy.forward(betas=[1.0])
    print("Toy finger, beta = 0 versus beta = 1. Joints are rows of J_regressor times vertices.")
    print("  DIP rest joint, beta 0:", mean["joints_rest"][3])
    print("  DIP rest joint, beta 1:", long["joints_rest"][3])
    delta = long["joints_rest"][3, 1] - mean["joints_rest"][3, 1]
    _check(np.isclose(delta, 0.4), "beta moves the DIP joint by the vertex's shape displacement")
    _check(np.allclose(mean["joints_rest"][0], 0.0), "beta 0 leaves the wrist joint on the template")
    print()
    print("On the real model the regressor is sparse and can contain negative weights,")
    print("so a joint need not lie inside the surface. It still moves when beta changes,")
    print("because the formula is J = J_regressor @ v_shaped, and v_shaped depends on beta.")
    print("Pose blend shapes are not part of this product. Shape moves the rest joints;")
    print("pose then rotates those rest joints.")


def lesson_6() -> None:
    _header(6)
    pose = np.zeros((4, 3))
    pose[2, 0] = np.deg2rad(90.0)
    spun = pose.copy()
    spun[0, 0] = np.deg2rad(30.0)
    feature = toy.pose_feature(pose)
    spun_feature = toy.pose_feature(spun)
    with_blend = toy.forward(pose)
    rigid = toy.forward(pose, use_pose_blend=False)
    print("PIP rotated 90 degrees about X. Pose feature length:", feature.size)
    print("  ||feature|| with that PIP rotation:", round(float(np.linalg.norm(feature)), 4))
    print("  feature changed by a wrist rotation?", not np.allclose(feature, spun_feature))
    gap = with_blend["v_posed"][4] - rigid["v_posed"][4]
    print("  pose-blend shift of the skin vertex:", np.round(gap, 4))
    _check(feature.size == 135 or feature.size == 27, "feature is 9 numbers per non-root joint")
    _check(np.allclose(feature, spun_feature), "global rotation is excluded from the pose feature")
    _check(np.isclose(gap[2], 0.05), "at 90 degrees the skin vertex gains 0.05 along Z before skinning")
    _check(np.allclose(toy.pose_feature(np.zeros(12)), 0.0), "the identity rotation has a zero pose blend")
    print()
    print("135 on the real model is 15 x 9. The root's 9 numbers are omitted on purpose:")
    print("turning the whole hand must not inflate the flesh correction.")


def lesson_7() -> None:
    _header(7)
    pose = np.zeros((4, 3))
    pose[2, 0] = np.deg2rad(90.0)
    out = toy.forward(pose, use_pose_blend=False)
    joints = out["joints"]
    vertices = out["vertices"]
    print("Rigid skinning (pose blend off), PIP at 90 degrees.")
    print("  DIP joint   ", np.round(joints[3], 4))
    print("  DIP vertex  ", np.round(vertices[3], 4))
    print("  PIP joint   ", np.round(joints[2], 4))
    print("  skin vertex ", np.round(vertices[4], 4), "  weights mcp=0.25 pip=0.75")
    _check(np.allclose(vertices[3], joints[3]), "a weight-1 vertex stays on its joint")
    _check(not np.allclose(vertices[4], joints[2], atol=1e-6), "a split weight does not lock to one bone")
    _check(np.allclose(WEIGHT_SUMS(), 1.0), "every LBS row sums to 1")
    print()
    print("The joint coordinates you want are the kinematic joints above.")
    print("Re-regressing joints from the skinned vertices gives a different point")
    print("as soon as a vertex is shared or a pose blend moves the surface.")


def WEIGHT_SUMS():
    return toy.WEIGHTS.sum(axis=1)


def lesson_8() -> None:
    _header(8)
    hands_mean = np.array([0.30, -0.20, 0.10])
    components = np.eye(3)
    pca_zero = hands_mean + np.zeros(3) @ components
    flat = np.zeros(3)
    coeffs_for_flat = flat - hands_mean
    print("A 3-D stand-in for the 45-D finger PCA:")
    print("  pose at PCA coefficients 0:", pca_zero, "  <- training-mean hand, slightly bent")
    print("  flat axis-angle pose:     ", flat)
    print("  PCA coefficients that reconstruct the flat pose:", coeffs_for_flat)
    _check(not np.allclose(pca_zero, 0.0), "PCA zero is not the flat template")
    _check(np.allclose(hands_mean + coeffs_for_flat @ components, flat), "flat pose = mean + coefficients @ basis")
    print()
    print("On the official model, flat_hand=True uses axis-angle directly and skips hands_mean.")
    print("flat_hand=False adds hands_mean. pose_is_pca=True multiplies coefficients by")
    print("hands_components first. Six components cannot pose every finger independently.")
    print("Use all 45 axis-angles when a joint angle has to be exact.")


def lesson_9() -> None:
    _header(9)
    print("Right-hand fingertip vertices:", sk.TIP_VERTEX_IDS)
    print("Sixteen joints plus these five vertices are the usual 21 surface points.")
    print()
    axis_angle = np.array([0.4, -0.7, 0.2])
    mirrored = axis_angle.copy()
    mirrored[1] *= -1
    mirrored[2] *= -1
    reflect = np.diag([-1.0, 1.0, 1.0])
    original = sk.rodrigues(axis_angle)
    reflected = reflect @ original @ reflect
    _check(np.allclose(sk.rodrigues(mirrored), reflected), "mirroring X flips the Y and Z axis-angle components")
    print("To mirror a right-hand pose across the YZ plane, negate components 1 and 2")
    print("of every joint's axis-angle, and negate the X of every point. Swap face")
    print("winding if you build triangles yourself. The release already contains")
    print("MANO_LEFT.pkl, so prefer that file over mirroring blend shapes by hand.")
    print()
    print("MANO does not have a fingertip hinge, a multi-bone wrist, or a muscle.")
    print("The pose blend is linear in (R - I). Extreme poses leave the training span.")


def lesson_10() -> None:
    _header(10)
    secret_deg = 40.0
    pose = np.zeros((4, 3))
    pose[2, 0] = np.deg2rad(secret_deg)
    target = toy.forward(pose)["vertices"][3]

    def loss(degree: float) -> float:
        trial = np.zeros((4, 3))
        trial[2, 0] = np.deg2rad(degree)
        tip = toy.forward(trial)["vertices"][3]
        return float(np.sum((tip - target) ** 2))

    found = minimize_scalar(loss, bounds=(-20.0, 120.0), method="bounded")
    print(f"Target DIP vertex came from an unknown PIP flexion.")
    print(f"  recovered angle: {found.x:.2f} degrees")
    print(f"  true angle:      {secret_deg:.2f} degrees")
    _check(abs(found.x - secret_deg) < 0.05, "one-parameter fit recovers the PIP angle")
    print()
    print("A real MANO fit is this same loop in more variables: global rotation,")
    print("finger pose or PCA coefficients, betas, and translation. The loss compares")
    print("joints or vertices to a target and usually penalizes large betas and large")
    print("PCA coefficients. Match fingertips to fingertip vertices, not to joint 15.")


def lesson_11() -> None:
    _header(11)
    try:
        model = require_model()
    except ManoModelMissing as exc:
        print(exc)
        return

    flat = model.forward(flat_hand=True)
    mean_pose = model.forward(flat_hand=False)
    betas = np.zeros(model.shapedirs.shape[-1])
    betas[0] = 1.0
    shaped = model.forward(betas=betas, flat_hand=True)
    weight_sums = model.weights.sum(axis=1)

    print(f"Loaded {model.path}")
    print(f"  vertices {model.v_template.shape}, faces {model.faces.shape}")
    print(f"  shapedirs {model.shapedirs.shape}, posedirs {model.posedirs.shape}")
    print(f"  weights {model.weights.shape}, J_regressor {model.j_regressor.shape}")
    wrist_to_middle = flat["joints"][4] - flat["joints"][0]
    print("  wrist -> middle MCP on the flat template (metres):", np.round(wrist_to_middle, 4))
    print("  That vector is the official template axis. Do not assume it is +Y.")
    moved = np.linalg.norm(shaped["joints"] - flat["joints"])
    mean_gap = np.linalg.norm(mean_pose["joints"] - flat["joints"])
    print(f"  joint motion from beta0=+1: {moved:.4f} m")
    print(f"  joint gap between flat axis-angle and hands_mean: {mean_gap:.4f} m")
    _check(model.v_template.shape == (778, 3), "template has 778 vertices")
    _check(model.faces.shape[0] == 1538, "template has 1538 faces")
    _check(np.allclose(weight_sums, 1.0, atol=1e-4), "LBS rows sum to 1")
    _check(moved > 1e-6, "a shape component moves the joints")
    _check(mean_gap > 1e-6, "hands_mean is not the flat pose")

    identity = model.forward(flat_hand=True)
    global_orient = np.array([0.3, -0.2, 0.1])
    turned = model.forward(global_orient=global_orient, flat_hand=True)
    rotation = sk.rodrigues(global_orient)
    recovered = (turned["vertices"] - turned["joints"][0]) @ rotation
    reference = identity["vertices"] - identity["joints"][0]
    _check(
        np.allclose(recovered, reference, atol=1e-5),
        "global rotation does not change shape, only the rigid pose",
    )
    print()
    print("Official forward pass matches the toy: shape, then joints, then pose blend, then LBS.")


LESSONS = {
    0: lesson_0,
    1: lesson_1,
    2: lesson_2,
    3: lesson_3,
    4: lesson_4,
    5: lesson_5,
    6: lesson_6,
    7: lesson_7,
    8: lesson_8,
    9: lesson_9,
    10: lesson_10,
    11: lesson_11,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MANO course lessons")
    parser.add_argument("lesson", nargs="?", type=int, help="lesson number 0-11; default is 0-10")
    args = parser.parse_args()
    if args.lesson is None:
        numbers = list(range(0, 11))
    else:
        if args.lesson not in LESSONS:
            raise SystemExit(f"Choose a lesson from 0 to 11. Got {args.lesson}.")
        numbers = [args.lesson]
    for number in numbers:
        LESSONS[number]()
    print()
    print("Textbook: COURSE.md")


if __name__ == "__main__":
    main()
