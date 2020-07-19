
from BG.BG_learning import learn_piecewise_subspace
from Alignment.Alignment_learning import learn_alignment


def main():

    # learn bg:
    learn_piecewise_subspace()

    # learn alignment
    learn_alignment()


if __name__ == '__main__':
    main()

