# MANO

A self-contained course on the MANO hand model. It does not depend on the rest of this repository.

Read the notebooks in order, starting at [tutorial/00_start_here.ipynb](tutorial/00_start_here.ipynb). The short notes are in [TUTORIAL.md](TUTORIAL.md), and the formula notes are in [COURSE.md](COURSE.md).

```bash
conda activate mano
cd MANO
python tutorial.py
```

The conda environment is `mano`, Python 3.11. Create it with `conda create -n mano python=3.11` and `pip install -r requirements.txt` if it is not already there.

| Command | What it does |
|---|---|
| `python tutorial.py` | 按 TUTORIAL.md Generalize the hand. |
| `python course.py` | Lessons 0–10. Skeleton, axis-angle, and the full forward equation on a 5-vertex finger. |
| `python course.py 11` | Same equation on `mano_v1_2/models/MANO_RIGHT.pkl`. |
| `python learn_mano.py` | Prints the 16-joint tree and writes `out/schematic_hand.png`. Bone lengths in that figure are a sketch. |

The official release in this folder is `mano_v1_2/`. `official_mano.py` reads `mano_v1_2/models/MANO_RIGHT.pkl`. The licence does not allow committing those pickles.

| File | Role |
|---|---|
| `COURSE.md` | The textbook. |
| `course.py` | Checks for each lesson. |
| `toy_mano.py` | One finger, same equations as MANO, no official weights. |
| `mano_skeleton.py` | Joint names, parents, schematic kinematics. |
| `official_mano.py` | Loader and forward pass for the real pickle. |
| `learn_mano.py` | Tree printout and the sketch figure. |
