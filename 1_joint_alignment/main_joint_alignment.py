
from SE.SE_alignment import SE_align
from STN.STN_run import main_STN
from AFFINE.AFFINE_global import get_global_AFFINE


def main():

    # run SE alignment
    SE_align()

    # run STN
    main_STN()

    # Prepare estimated transformations (x_i, T^theta_i)
    get_global_AFFINE()

    print('\nFinished Module 1.')

if __name__ == '__main__':
    main()

