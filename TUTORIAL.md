# MANO 教程：从 0 到能自己调用

按章读。每一章末尾有一条命令，在 `mano` 环境里跑，对照终端和图片。不要跳着只看图。

```bash
conda activate mano
cd MANO
```

模型文件已经在 `mano_v1_2/models/MANO_RIGHT.pkl`。这是 MPI 发布的右手，778 个顶点、1538 个三角面、16 个关节，单位米。论文是 Romero, Tzionas, Black，*Embodied Hands*，SIGGRAPH Asia 2017。MANO 就是 SMPL 用在手上。

学完你应该能不看笔记做完这些事：写出任意一个关节在姿态向量里的下标，加载模型，弯曲指定的手指，改变手的形状，取出 16 个关节和 5 个指尖，并说清楚为什么全零的 PCA 系数不是伸直的手。

## 1. 先建立这一张图

一次前向计算吃进四个量，吐出网格和关节：

| 输入 | 长度 | 它改变什么 |
|---|---|---|
| `global_orient` | 3 | 整只手怎么转。这就是手腕。 |
| `hand_pose` | 45，或者更少的 PCA 系数 | 15 个手指关节怎么转。 |
| `betas` | 10 | 这是谁的手：长短、胖瘦。不决定握拳还是张开。 |
| `transl` | 3 | 手在哪里，单位米。 |

输出是 778 个顶点。16 个关节不是另一套骨骼文件，而是用一个 16×778 的矩阵从顶点里算出来的。

公式后面第 8 章再拆开。现在只要记住：形状先改网格，关节从改过的网格里回归出来，姿态再去转这些关节，蒙皮让网格跟着骨头走。

## 2. 背下这 16 个关节

顺序不是从拇指到小指。小指排在无名指前面。拇指的第 1 节是腕掌关节 CMC，不是指根的掌指关节。

| 下标 | 名字 | 解剖 | 父关节 | 48 维姿态里的切片 |
|---|---|---|---|---|
| 0 | wrist | 腕 | 无 | `[0:3]`，全局旋转 |
| 1–3 | index1/2/3 | 食指 MCP、PIP、DIP | 0 → 1 → 2 | `[3:12]` |
| 4–6 | middle1/2/3 | 中指 MCP、PIP、DIP | 0 → 4 → 5 | `[12:21]` |
| 7–9 | pinky1/2/3 | 小指 MCP、PIP、DIP | 0 → 7 → 8 | `[21:30]` |
| 10–12 | ring1/2/3 | 无名指 MCP、PIP、DIP | 0 → 10 → 11 | `[30:39]` |
| 13–15 | thumb1/2/3 | 拇指 CMC、MCP、IP | 0 → 13 → 14 | `[39:48]` |

`hand_pose` 把第 0 节拿掉，所以只剩 45 个数。关节 `i`（从 1 数到 15）占用

```text
hand_pose[3*(i-1) : 3*(i-1)+3]
```

没有第 17 个关节。指尖不在这棵树上。

```bash
python tutorial.py 2
```

终端会把树打出来。对着表核对：7 是小指，10 是无名指。

## 3. 轴角，以及这只官方手的屈曲轴

每一节是 3 个弧度 `(rx, ry, rz)`，方向是轴，长度是角度。零向量表示不转。

休息姿态下，关节自己的坐标轴和模型坐标轴重合，并不沿着骨头。所以「屈曲」是绕哪根轴，要看这只模板里手指指向哪里，不能从解剖图直接抄。

这只右手的伸直模板里，手指指向 **-X**，掌心朝 **-Y**，五指在 **Z** 方向上张开，拇指在 **+Z**。因此：

- 绕局部 **Z** 的正角度：四指弯向掌心
- 绕局部 **Y**：手指向两侧分开或并拢
- 绕局部 **X**：大致是沿手指扭转，指尖几乎不动

教学用的 `learn_mano.py` 示意图把手指画成 +Y，那张图只用来看树，不能拿去转官方模型。

一个关节自己的旋转不移动它自己，只移动它的子关节和皮肤。食指 PIP 转了，食指 MCP 和手腕不动。

## 4. 看官方右手长什么样

```bash
python tutorial.py 4
```

打开 `out/tutorial/04_flat_and_mean.png`。两只手都从掌心一侧看（从 -Y 看过去）。红点是五个指尖顶点。

左图：`betas = 0`，手指轴角全 0。这是注册时的伸直模板。

右图：同一只手，加上文件里的 `hands_mean`。手指收拢，拇指转到掌心前。这是训练集里的平均姿态。

终端会打印腕到中指 MCP 的向量，大约是 `(-0.095, -0.002, -0.003)` 米。第一个数最大，所以手指沿 -X。食指尖从伸直到平均姿态，Y 减小大约 6 厘米，所以掌心在 -Y。

同一张对比也在 `out/official_right_hand.png`，那里有手背和掌心各一面。

## 5. 自己算一次

在 Python 里：

```python
from official_mano import require_model
import numpy as np

model = require_model()
flat = model.forward(flat_hand=True)
print(flat["vertices"].shape)   # (778, 3)
print(flat["joints"].shape)     # (16, 3)
print(flat["tips"]["index"])    # 食指尖，米
```

`require_model()` 读的就是 `mano_v1_2/models/MANO_RIGHT.pkl`。`forward` 不传参数时，全局旋转、手指姿态、形状、平移都是 0，并且 `flat_hand=True`，所以结果就是伸直模板。

关节坐标在 `flat["joints"][i]`，名字顺序就是第 2 章的表。

## 6. 只弯曲食指

食指三节是 `hand_pose` 的前 9 个数。每节的第三个分量是绕 Z。

```bash
python tutorial.py 6
```

看 `out/tutorial/06_index_curl.png`。右图只把食指 MCP、PIP、DIP 绕 Z 转了 40°、65°、45°。食指尖移动大约 10 厘米，另外四指的指尖移动不到 0.1 毫米。

同样的角度如果绕 X，食指尖只移动大约 4 毫米。这就是第 3 章说的扭转。

要改角度，编辑 `tutorial.py` 里 `chapter_6` 的 `np.deg2rad([40.0, 65.0, 45.0])`，再跑一次。

## 7. 形状：betas

`betas` 全 0 是训练扫描的平均手形。第 0 个系数是变化最大的那一维，通常和手的整体尺度有关。

```bash
python tutorial.py 7
```

`out/tutorial/07_shape.png` 是 `beta[0]` 取 -2、0、+2，姿态都是伸直。腕到中指尖大约是 192 mm、175 mm、159 mm。手还是张开的。握拳必须改 `hand_pose`，不能靠 `betas`。

文件里实际存了 20 个形状基，发布模型通过一个下标选择只用前 10 个。`model.shapedirs.shape` 是 `(778, 3, 10)`。

## 8. 把前向公式拆开

```text
v_shaped = v_template + shapedirs · betas
J        = J_regressor · v_shaped
v_posed  = v_shaped + posedirs · (R - I)
vertices = 线性蒙皮(v_posed, J, pose, weights)
joints   = 沿父子树做正运动学
```

对照文件里的数组：

| 名字 | 形状 | 含义 |
|---|---|---|
| `v_template` | (778, 3) | 平均手，米 |
| `f` | (1538, 3) | 三角面 |
| `shapedirs` | (778, 3, 10) | 10 个形状基 |
| `posedirs` | (778, 3, 135) | 15 个关节 × 每个旋转矩阵的 9 个数 |
| `J_regressor` | (16, 778) | 关节是顶点的线性组合，权重可以是负数 |
| `weights` | (778, 16) | 每个顶点蒙在哪些骨头上，每一行加起来是 1 |
| `kintree_table` | (2, 16) | 第 0 行是父关节。根的父节点在文件里常常写成 4294967295，也就是 -1 |
| `hands_components` | (45, 45) | 手指姿态的 PCA 基 |
| `hands_mean` | (45,) | 训练集的平均手指姿态 |

`R - I` 只对 15 个手指关节计算，拼成 135 个数。手腕的旋转不放进去。否则把整只手在空中转一圈，肉的修正量也会变。全是单位旋转时，`R - I` 是 0，伸直模板不会凭空鼓起一块。

关节用的是形状混合之后的顶点，不用姿态混合之后的顶点。所以 `betas` 会挪动关节的休息位置，姿态混合不会。

你要的关节坐标是正运动学算出来的那 16 个点。不要在蒙皮之后的表面上再回归一次。顶点一旦被两根骨头共享，这两个结果就不一样。

这三步在真手上是同一段代码。用 5 个顶点的玩具手指可以把每一步看完，公式和上面相同：

```bash
python course.py 5
python course.py 6
python course.py 7
```

`course.py` 的输出是英文检查。过了就说明式子和实现一致。

## 9. 两个「全零」

这是最容易用错的地方。

1. `flat_hand=True`，`hand_pose` 全 0：45 个轴角就是 0，手是伸直模板。
2. `flat_hand=False`，`hand_pose` 全 0：先加上 `hands_mean`，手是训练集的平均姿态，也就是第 4 章右图。

PCA 系数全 0 属于第二种。低维系数先乘 `hands_components`，再视情况加 `hands_mean`：

```text
手指轴角 = hands_mean + 系数 @ hands_components[:n]
```

```bash
python tutorial.py 9
```

终端里两只手的手腕坐标相同，关节整体却差大约 10 厘米。`hands_mean` 里没有全局旋转，腕不动，弯的是手指。

6 个 PCA 系数只能表示很光滑的一类手势，不能把每个关节转到指定角度。要精确的关节角，就用 45 维轴角，并设 `flat_hand=True`。

```python
out = model.forward(hand_pose=coeffs, flat_hand=False, pose_is_pca=True)
```

## 10. 指尖、左手、模型做不到的事

```bash
python tutorial.py 10
```

五个指尖是网格顶点，不是关节：

| 手指 | 顶点号 |
|---|---|
| 拇指 | 744 |
| 食指 | 320 |
| 中指 | 443 |
| 无名指 | 554 |
| 小指 | 671 |

平时说的 21 个点 = 上面 16 个关节 + 这 5 个顶点。关节 15 是拇指的 IP 关节。拿它去对拇指尖，会对错将近两厘米。

左手用 `mano_v1_2/models/MANO_LEFT.pkl`，不要自己去镜像混合形状。如果只是把右手的姿态镜像过去：每个轴角的 Y、Z 分量变号，X 分量不变；点的 X 坐标变号。`course.py` 第 9 课检查了这个符号。

MANO 没有单独的指尖铰链，手腕也只是一个关节，没有一排腕骨。姿态混合是 `R - I` 的线性函数，离开训练姿态很远时网格会变假。

## 11. 和深度图评测里的关节不是同一套

[awesome-hand-pose-estimation 的 evaluation](https://github.com/xinghaochen/awesome-hand-pose-estimation/tree/master/evaluation) 评的是 NYU、ICVL、MSRA 上预测点和标注点的欧氏距离，单位毫米。那些标注是像素加深度，再变到相机坐标。它不读 MANO 的 48 维姿态。

| 数据 | 评测用几点 | 和 MANO 的差别 |
|---|---|---|
| ICVL | 16 | 每指只有根、中、尖，有掌心点，没有 DIP |
| NYU | 36 个里只用 14 个 | 下标是 0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32 |
| MSRA | 21 | 腕，然后食指到拇指，每指 MCP、PIP、DIP、指尖 |

MSRA 也是 21 个点，最容易和「MANO 的 16 关节加 5 个指尖」搞混。顺序、指尖的定义、腕的定义都不一样。要对齐，先把 MANO 关节和那 5 个顶点排成对方的顺序，再变到同一个相机坐标系，然后才用毫米误差。

## 12. 最小调用，收在一起

```python
import numpy as np
from official_mano import require_model

model = require_model()

hand_pose = np.zeros(45)
hand_pose[2] = np.deg2rad(40)   # 食指 MCP，绕 Z，弯向掌心
hand_pose[5] = np.deg2rad(65)   # 食指 PIP
hand_pose[8] = np.deg2rad(45)   # 食指 DIP

betas = np.zeros(10)

out = model.forward(
    global_orient=np.zeros(3),
    hand_pose=hand_pose,
    betas=betas,
    transl=np.zeros(3),
    flat_hand=True,
)

vertices = out["vertices"]          # (778, 3) 米
joints = out["joints"]              # (16, 3)，下标见第 2 章
index_tip = out["tips"]["index"]    # 顶点 320
faces = model.faces                 # (1538, 3)
```

全局旋转是另外 3 个数，加在 48 维姿态的最前面。它不进 `hand_pose`，也不进姿态混合。

## 你现在应该能答上来

先盖住答案。

1. 下标 7 和 10 各是哪根手指的哪一节？
2. 食指 PIP 在 `hand_pose` 的哪三个位置？绕哪根轴弯向这只右手的掌心？
3. `betas` 全 0、`hand_pose` 全 0、`flat_hand=False` 得到的是伸直手吗？
4. 指尖一共有几个关节？拇指尖的顶点号是多少？
5. 为什么转手腕不会改变 `posedirs` 用的那 135 个数？
6. MSRA 的 21 个点能不能按 MANO 的 0 到 15 直接相减？

答案：

1. 7 是小指 MCP，10 是无名指 MCP。
2. `hand_pose[3:6]`。这只官方右手绕局部 Z 的正角度弯向掌心（-Y）。
3. 不是。那是 `hands_mean`，手指是弯的。伸直模板要 `flat_hand=True`。
4. 指尖没有关节。拇指尖是顶点 744。
5. 那 135 个数来自 15 个手指关节的 `R - I`。手腕是根，被排除在外，这样整手转动不会改变肉的修正。
6. 不能。MSRA 含指尖，顺序是食指、中指、无名指、小指、拇指。MANO 的 16 个关节没有指尖，而且小指在无名指前面。
