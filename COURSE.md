# MANO from zero

Read one lesson, then run it. The checks print `check passed` when the fact is true in code.

```bash
conda activate mano
cd MANO
python course.py        # lessons 0-10, no official file required
python course.py 6      # one lesson
python course.py 11     # reads mano_v1_2/models/MANO_RIGHT.pkl
python learn_mano.py    # drawing of the 16-joint tree
```

Paper: Romero, Tzionas, Black, *Embodied Hands: Modeling and Capturing Hands and Bodies Together*, SIGGRAPH Asia 2017. MANO is the hand formulation of SMPL. The released model is a right hand and a left hand, each with 778 vertices and 1538 faces.

Lessons 0-10 are the whole algorithm on a 5-vertex finger that uses the same equations. Lesson 11 reads the release at `mano_v1_2/models/MANO_RIGHT.pkl`. The pickle stores 20 shape components and exposes the first 10.

## 0. The function

One forward pass:

```
v_shaped = v_template + shapedirs · betas
J        = J_regressor · v_shaped
v_posed  = v_shaped + posedirs · (R(pose) - I)      # non-root joints only
vertices = LBS(v_posed, J, pose, weights)
joints   = FK(J, pose)
```

You pass in four things:

| Name | Size | Role |
|---|---|---|
| `global_orient` | 3 | Axis-angle of the wrist. Rotates the whole hand. |
| `hand_pose` | 45, or fewer PCA coefficients | Axis-angle of the 15 finger joints. |
| `betas` | usually 10 | Identity of the hand: longer, wider, thicker. Not fist versus palm. |
| `transl` | 3 | Translation in metres. |

`global_orient` together with `hand_pose` is the 48-D pose. Joint `i` uses entries `[3i : 3i+3]`.

## 1. Memorize the tree

Pinky is stored before ring. Thumb1 is the carpometacarpal joint.

| Index | Name | Anatomy | Parent | Pose slice |
|---|---|---|---|---|
| 0 | wrist | wrist | none | `[0:3]` global |
| 1, 2, 3 | index1/2/3 | MCP, PIP, DIP | 0, 1, 2 | `[3:12]` |
| 4, 5, 6 | middle1/2/3 | MCP, PIP, DIP | 0, 4, 5 | `[12:21]` |
| 7, 8, 9 | pinky1/2/3 | MCP, PIP, DIP | 0, 7, 8 | `[21:30]` |
| 10, 11, 12 | ring1/2/3 | MCP, PIP, DIP | 0, 10, 11 | `[30:39]` |
| 13, 14, 15 | thumb1/2/3 | CMC, MCP, IP | 0, 13, 14 | `[39:48]` |

There is no fingertip joint. The usual 21 points are these 16 joints plus five vertices: thumb 744, index 320, middle 443, ring 554, pinky 671.

`python course.py 1` prints the tree and checks the pinky/ring trap.

## 2. Axis-angle

Each joint is three radians `(rx, ry, rz)`. The direction is the axis, the length is the angle. The Rodrigues formula turns that into a 3×3 matrix. The zero vector is the identity.

At rest, the joint coordinate frame is aligned with the model axes. It is not a frame whose X axis runs along the bone. On the teaching skeleton only, the four fingers point along +Y, the palm faces +Z, and a positive rotation about X flexes the finger toward the palm. The official template has its own orientation. Lesson 11 prints the wrist-to-middle-MCP vector so you can see it instead of assuming +Y.

## 3. Forward kinematics

Rest joints come from step `J` above. Joint 0 sits at `J[0]`. For every other joint:

```
R_i     = R_parent @ Rodrigues(pose_i)
offset  = J[i] - J[parent]
p_i     = p_parent + R_parent @ offset
```

The joint's own rotation is applied after its position is computed, so it moves children and skin, not itself. A finger rotation does not move the wrist. `python course.py 3` rotates index PIP by 40 degrees and checks those two facts.

`learn_mano.py` draws the same tree with invented bone lengths. Use it to see the connectivity. Do not cite those millimetres as MANO.

## 4. The pickle

`MANO_RIGHT.pkl` is a dictionary saved with chumpy. The arrays that the formula needs:

| Key | Shape | Meaning |
|---|---|---|
| `v_template` | (778, 3) | Mean registered hand, metres |
| `f` | (1538, 3) | Faces |
| `shapedirs` | (778, 3, 10) | Shape basis, one column per beta |
| `posedirs` | (778, 3, 135) | Pose corrective basis |
| `J_regressor` | (16, 778) | Sparse linear map, negative weights allowed |
| `weights` | (778, 16) | Skinning weights, rows sum to 1 |
| `kintree_table` | (2, 16) | Row 0 is the parent. The root parent is often 4294967295 |
| `hands_components` | (45, 45) | PCA basis of finger axis-angles |
| `hands_mean` | (45,) | Mean finger pose of the training registrations |

`hands_coeffs` is the PCA spectrum. You do not multiply by it during the forward pass.

## 5. Shape

```
v_shaped = v_template + sum_k beta_k * shapedirs[:, :, k]
J        = J_regressor @ v_shaped
```

Beta 0 is the mean shape of the training scans. Changing beta moves vertices and therefore moves joints. Pose blend shapes are not inside `J`. The toy finger in `toy_mano.py` has one beta that lengthens the part past the MCP by 0.2 at the PIP vertex and 0.4 at the DIP vertex, and the DIP joint moves by exactly 0.4 because that joint is defined as that vertex.

On the real regressor a joint is a signed mix of several vertices, so it can sit off the surface. The dependence on beta is the same product.

## 6. Pose blend shapes

Take the 15 finger rotations, skip the wrist, and stack `R - I` in row-major order. That is a 135-vector. It is zero when every finger rotation is the identity, which is why the flat pose does not get a fake correction.

```
v_posed = v_shaped + posedirs · pose_feature
```

The wrist rotation is omitted. If it were included, spinning the hand in the air would change the flesh correction. Lesson 6 checks that a wrist rotation leaves the feature untouched, and that a 90-degree PIP rotation on the toy moves the PIP vertex by `(0, 0, 0.05)` before skinning.

This term is how MANO gets deformable flesh out of a rigid skeleton. It is linear in `R - I`. It is not a muscle model, and it gets worse far from the poses used to build `posedirs`.

## 7. Skinning

Build a global rotation and a position for every joint with the kinematic tree. The 4×4 that takes a rest-pose point into the posed hand is

```
T_i = [ R_i ,  p_i - R_i @ J_i ]
      [ 0   ,  1            ]
```

Then

```
vertices = sum_i weight[vertex, i] * T_i * v_posed
```

A vertex whose row is a single 1 stays glued to that joint. A vertex with weight split across two bones lands between them. Lesson 7 puts weights `(0, 0.25, 0.75, 0)` on a vertex that sits off the PIP joint, and checks both cases. The vertex has to sit off the joint: a point exactly on the hinge does not move when that hinge rotates, so the blend is invisible.

Report the kinematic joints from `FK`, not a second regression on the skinned vertices. Those two sets diverge as soon as a weight is shared or a pose blend moves a vertex.

## 8. Two zeros

Axis-angle zero, with `flat_hand=True`, is the template pose: fingers in the registered flat configuration, pose blend equal to zero.

PCA coefficient zero is different. The release stores a low-dimensional basis of the 45 finger angles:

```
finger_axis_angle = hands_mean + coefficients @ hands_components[:n]
```

`hands_mean` is not the zero vector, so PCA zeros give the average training pose, which is slightly bent. To hit the flat template in PCA space you need the coefficients that cancel `hands_mean`. `official_mano.py` exposes this as `flat_hand` and `pose_is_pca`.

Six PCA components are a smooth prior. They cannot put each joint on an arbitrary angle. Use 45 axis-angles when the angle itself is the quantity you care about.

## 9. Left hand

Reflecting a rotation across the YZ plane (`x -> -x`) keeps the X component of the axis-angle and negates Y and Z. Lesson 9 checks `S R S` against that sign rule. Also negate X of translations and vertex coordinates, and swap triangle winding.

`MANO_LEFT.pkl` is already the mirrored model, including blend shapes. Load it rather than mirroring `posedirs` yourself.

MANO still will not give you an independent fingertip hinge or a carpal stack. The wrist is one joint.

## 10. Using it

Generate:

```python
from official_mano import OfficialMano
model = OfficialMano("mano_v1_2/models/MANO_RIGHT.pkl")
out = model.forward(global_orient, hand_pose, betas, transl, flat_hand=True)
# out["vertices"] (778, 3), out["joints"] (16, 3), out["tips"] dict of 5 points
```

Fit: choose pose, betas, and translation so joints or vertices approach a target, and usually penalize large betas and large PCA coefficients. Lesson 10 recovers a hidden PIP angle on the toy finger by minimizing the DIP-vertex error. The same loop on the real model has 48 + 10 + 3 numbers instead of one. Compare a fingertip target with `out["tips"]`, not with joint 15. Joint 15 is the thumb IP joint.

## 11. Official file

After `mano_v1_2/models/MANO_RIGHT.pkl` is present:

```bash
python course.py 11
python learn_mano.py official
```

Lesson 11 checks 778 vertices, 1538 faces, skinning rows summing to 1, a nonzero effect of `beta[0]`, a nonzero gap between the flat pose and `hands_mean`, and that a pure global rotation is a rigid motion. On this release the flat-template vector from the wrist to the middle MCP is about `(-0.095, -0.002, -0.003)` metres, so the fingers point along **-X**. The teaching sketch's +Y axis was only a drawing.

## You can reconstruct the model when these are automatic

- Index 7 is the pinky MCP. Index 10 is the ring MCP. Thumb1 is the CMC.
- Pose of joint `i` is three radians at `[3i:3i+3]`. Joint 0 is global and is outside the pose blend and outside the PCA.
- A joint rotation does not move that joint. It moves descendants and the skin.
- `J = J_regressor @ v_shaped`. Betas move joints. Pose blend shapes do not define the rest joints.
- `posedirs` sees a 135-vector. The wrist is not in it.
- LBS weights are per vertex and sum to 1. Kinematic joints and re-regressed surface points are different objects.
- PCA zero is `hands_mean`. Axis-angle zero is the flat template.
- Fingertips are vertices 744, 320, 443, 554, 671.
- A left-hand pose mirrored from the right negates the Y and Z parts of every axis-angle.
