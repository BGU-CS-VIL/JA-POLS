

# JA-POLS

**Authors:** Irit Chelly, Vlad Winter, Dor Litvak, David Rosen, and Oren Freifeld.

This code repository corresponds to our CVPR-20 paper: **JA-POLS: a Moving-camera Background Model via Joint Alignment and Partially-overlapping Local Subspaces**.
JA-POLS is a novel 2D-based method for unsupervised learning of a moving-camera background model, which is highly scalable and allows for relatively-free camera motion.
<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/tennis_gif.gif" alt="JA-POLS typical results" width="520" height="320">
</p>

## Requirements
- Python: most of the code runs in python using tensorflow, openCV, scikit-image, and other common python packages.
- MATLAB (for the SE-Sync part)
- C++: in case you are choosing the TGA mathod for learning the local subspaces (see module 2 below), please follow the requirements [here](https://github.com/MPI-IS/Grassmann-Averages-PCA). All steps should be performed in the TGA folder: *2_learning\BG\TGA-PCA*.


**For a minimal working example, use the Tennis sequence (the input images are already located in the input folder in this repository)**.

## Installation

## Instructions and Description
JA-POLS method includes 3 phases that run in separate modules:
- Joint alignment: align all input images to a common coordinate system
- Learning of two tasks:
    - Partially-overlapping Local Subspaces (the background)
    - Alignment prediction
- BG/FG separation for a (previously-unseen) input frame 

**Configuration parameters:** the file config.py includes all required parameters for the 3 modules.

Before start running the code, insert the following config parameter:

Your local path to the JA-POLS folder:
```
paths = dict(
    my_path = '/PATH_TO_JAPOLS_CODE/JA-POLS/',
)
```

The size of a single input frame (height, width, depth):
```
images = dict(
    img_sz = (250, 420, 3),
)
```

### Module 1: Joint Alignment
<ins>Code</ins>:<br />
Main function: *1_joint_alignment/main_joint_alignment.py*

<ins>Input</ins>:<br />
A video or a sequence of images, that the BG model will be learned from.<br />
The video or the images should be located in *input/learning/video* or *input/learning/images* respectively.

<ins>Output</ins>:<br />
- *data/final_AFFINE_trans.npy*: affine transformations for all input images.<br />
(In this file, record *i* contains the affine transformation (6-parameters vector) that is associated with input image *i*).

<ins>Required params in config.py:</ins><br />
Data type (video or a sequence of images), and relevant info about the input data:
```
se = dict(
    data_type = 'images',  # choose from: ['images', 'video']
    video_name = 'jitter.mp4',  # relevant when data_type = 'video'
    img_type = '*.png',  # relevant when data_type = 'images'
)
```

Parameters for the spatial transformer net (when estimating the affine transformations):
```
stn = dict(
    device = '/gpu:0',   # choose from: ['/gpu:0', '/gpu:1', '/cpu:0']
    load_model = False,  # 'False' when learning a model from scratch, 'True' when using a trained network's model
    iter_per_epoch = 2000, # number of iterations 
    batch_size = 10,
)
```

The rest of the parameters can (optionally) remain with the current configuration.

<ins>Description</ins>:<br />
Here we solve a joint-alignment problem: 

<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/joint_align_0.png" alt=" " width="520" height="290">
</p>

<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/joint_align_desc2.png" alt=" " width="680" height="450">
</p>

<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/se_graph.png" alt=" " width="280" height="100">
</p>

High-level steps:
1. Compute relative transformations for pairs of input images (according to the graph)
2. Run SE-Sync framework and get absolute SE transformations for each frame
3. Transform images according to the absolute SE transformations
4. Estimate residual affine transformations by optimizing the above loss function using Spatial Transformer Network (STN).
5. End-up with absolute affine transformations for each of the input images
<br />
<br />

### Module 2: Learning
<ins>Code location (main function)</ins>:<br /> 
Main function: *2_learning/main_learning.py*

<ins>Input</ins>:<br /> 
Files that were prepared in module 1:
- *data/final_AFFINE_trans.npy*
- *data/imgs.npy*
- *data/imgs_big_embd.npy*

<ins>Output</ins>:<br />
- *data/subspaces/*: local subspaces for the background learning.<br /> 
- *2_learning/Alignment/models/best_model.pt*: model of a trained net for the alignment prediction.

<ins>Required params in config.py:</ins><br />
**Local-subspaces learning:**<br />
Method type of the background learning algorithm, that will run on each local domain:
```
pols = dict(
    method_type = 'PRPCA',  # choose from: [PCA / RPCA-CANDES / TGA / PRPCA]
)
```
The rest of the parameters can (optionally) remain with the current configuration.

**Alignment-prediction learning:**<br />
Parameters for the regressor net (when learning a map between images and transformations):
```
regress_trans = dict(
    load_model = False,  # 'False' when learning a model from scratch, 'True' when using a trained network's model
    gpu_num = 0,  # number of gpu to use (in case there is more than one)
)
```
The rest of the parameters can (optionally) remain with the current configuration.


<ins>Description</ins>:<br />
Here we learn two tasks, based on the affine transformations that were learned in module 1:
<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/phase2.png" alt=" " width="270" height="240">
</p>

<br>
<p align="center">
<img src="https://github.com/itohamy/JA-POLS_v0/blob/master/tmp/learning_desc.png" alt=" " width="650" height="720">
</p>


### Module 3: Background/Foreground Separation
<ins>Code</ins>:<br />
Main function: *3_bg_separation/main_bg_separation.py*

<ins>Input</ins>:<br />
A video or a sequence of test images for BG/FG separation.<br />
The video or the images should be located in *input/test/video* or *input/test/images* respectively.

<ins>Output</ins>:<br />
- *output/bg/*: background for each test image.<br /> 
- *output/fg/*: foreground for each test image.<br />
- *output/img/*: original test images.<br /> 

<ins>Required params in config.py:</ins><br />
Data type (video or a sequence of test images), and relevant info about the input data:
```
bg_tool = dict(
    data_type = 'images',  # choose from: ['images', 'video']
    video_name = 'jitter.mp4',  # relevant when data_type = 'video'
    img_type = '*.png',  # relevant when data_type = 'images'
)
```

Indicate which test images to process: 'all' (all test data), 'subsequence' (subsequence of the image list), or 'idx_list' (a list of specific frame indices (0-based))..<br /> 
If choosing 'subsequence', insert relevant info in "start_frame" and "num_of_frames".<br /> 
If choosing 'idx_list', insert a list of indices in "idx_list".
```
bg_tool = dict(
    which_test_frames='idx_list',  # choose from: ['all', 'subsequence', 'idx_list']
    start_frame=0,
    num_of_frames=20,
    idx_list=(2,15,39),
)
```

Indicate whether or not to use the ground-truth transformations, in case your process images from the original video.<br />
When processing learning images, insert True.<br /> 
When processing unseen images, insert False.
```
bg_tool = dict(
    use_gt_theta = True,
)
```
The rest of the parameters can (optionally) remain with the current configuration.
