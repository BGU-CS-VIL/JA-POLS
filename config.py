

paths = dict(
    my_path = '/Users/irita/Documents/Master/Research/Tracking/Moving Camera - code/JA-POLS_v0/',
)


images = dict(
    img_sz = (250, 420, 3),
)


se = dict(
    data_type = 'images',  # choose from: ['images', 'video']
    video_name = 'jitter.mp4',  # relevant when data_type = 'video'
    img_type = '*.png',  # relevant when data_type = 'images'
)


stn = dict(
    device = '/gpu:0',   # choose from: ['/gpu:0', '/gpu:1', '/cpu:0']
    load_model = False,  # 'False' when learning a model from scratch, 'True' when using a trained network's model
    iter_per_epoch = 2000, # number of iterations
    batch_size = 10,

    num_stn = 3,
    weight_stddev = 1e-5,
    activation_func = "relu",  # "tanh" ,"relu"
    delta = 0.025,
    sigma = 0.7, # for Geman-Mecclure
    align_w = 1,
    regul_w = 1,
    regul_w2 = 0,
    alignment_reg = 10000,
    regulator_reg = 100,
    first_regulator = 'SE',
    second_regulator = 'SIMPLE',
    lrn_rate = 1e-5,
    ordered_batch = False,
)


pols = dict(
    method_type = 'TGA',  # choose from: [PCA / RPCA-CANDES / TGA / PRPCA]

    shift_sz = 40,  # stride used to split the big domain into local domains
    window_sz = (250, 420),  # window size used to split the big domain into local domains
    k = 5,  # number of learned components in each subspace, relevant for: pca, TGA, empca
    trimming_percent = 95,  # relevant for: TGA

    overlap_percent = 0.6, # minimum % of overlapped pixels out of d_tilde needed to consider an overlapped image (used in "get_overlapped_imgs")
    min_data_points = 5,  # minimum number of images in the dataset of each local subspace
    # Comment: if we want to learn the whole panorama (global model): overlap_percent=0, window_sz=img_emb_sz
)


regress_trans = dict(
    load_model = False, # 'False' when learning a model from scratch, 'True' when using a trained network's model
    gpu_num = 0, # number of gpu to use (in case there is more than one)
    model_name = 'googlenet',  # Models to choose from [googlenet, resnet, alexnet, vgg, squeezenet, densenet, inception]
    batch_size = 64,
    num_epochs = 1,  #200
    learning_rate = 0.001,
    val_prct = 0.2,

    # If augmentations are needed, set to True, and choose # of augmented images per input image
    augment = True,
    aug_num_per_img = 5,
    aug_std=0.1,

    class_num = 6,  # Fixed
    test_name = 'test_false_aug',
    feature_extract = False,  # Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
)


bg_tool = dict(
    data_type = 'images',  # choose from: ['images', 'video']
    video_name = 'jitter.mp4',  # relevant when data_type = 'video'
    img_type = '*.png',  # relevant when data_type = 'images'

    # Choose the number of test frames to process: 'all' (all test data), 'subsequence' (subsequence of the image list), or 'idx_list' (a list of specific frame indices).
    # If choosing 'subsequence': insert relevant values in "start_frame" and "num_of_frames".
    # If choosing 'idx_list': insert a list of indices in "idx_list".
    which_test_frames='idx_list',  # choose from: ['all', 'subsequence', 'idx_list']
    start_frame=0,
    num_of_frames=20,
    idx_list=(2,15,39),

    # use ground-truth transformations:
        # When processing learning images: insert True
        # When processing unseen images: insert False
    use_gt_theta = True,

    gpu_num=0,  # gpu to use when running the regression net.

    # refinement of the predicted alignment:
    only_refine = False,   # means that there is only SIFT refinement, where the image is placed in the center and warped towards the panorama. If this is True, a big 'gap_refine' is needed.
    gap_refine = 100,  # in the refinement process, this is the number of pixels gap we look at in the panorama, around the enclosing square.

    overlap_percent = 0.7, # minimum % of overlapped pixels out of (window_sz*window_sz*3) needed to consider a subspace to be overlapped (used in "run_bg_model")

    # whether to project on the whole big domain, or use overlapping local domains:
    is_global_model = False, # Comment: if we want to project on the whole panorama (global model): is_global_model=True, overlap_percent=0 (and in pols: window_sz=img_emb_sz)
)
