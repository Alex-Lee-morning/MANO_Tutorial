The release you downloaded already lives one level up:

    mano_v1_2/models/MANO_RIGHT.pkl
    mano_v1_2/models/MANO_LEFT.pkl

official_mano.py reads those paths. The licence does not allow the pickles
to be committed. .gitignore covers mano_v1_2/models/*.pkl.

With the conda env active, from the MANO directory:

    python course.py 11
    python learn_mano.py official
