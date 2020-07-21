import sys
sys.path.append('..')
sys.path.append('.')

from BG.Piecewise_Subspace import Piecewise_Subspace


def learn_piecewise_subspace():

    print('Start learning piecewise subspaces...')

    PS_sub = Piecewise_Subspace()

    # Prepare final transformations:
    PS_sub.prepare_image_transformations()

    # Learn local subspace:
    PS_sub.run_PS_subspace()


