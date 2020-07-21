
from BG_Tool import BG_Tool

def main():

    BG = BG_Tool()

    # Load test images to BG_Tool
    BG.load_test_images()

    # Get a (ground-truth / predicted) transformation for each of the test images:
    BG.get_theta_for_test_imgs()

    # Refine test thetas:
    BG.prepare_test_image_transformations_refined()

    # Run BG model on all test images (for debug, run on the training images):
    BG.run_bg_model()

    # Create a video from all test images (bg and fg):
    BG.create_bg_fg_video()

    print('\nFinish running BG tool.')

if __name__ == '__main__':
    main()


