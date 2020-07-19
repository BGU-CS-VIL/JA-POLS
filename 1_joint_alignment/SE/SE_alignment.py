import config
from SE.data_provider import DataProvider
from SE.graph_manager import GraphManager
import matlab.engine


def SE_align():

    mypath = config.paths['my_path']
    sesync_path = mypath + '1_joint_alignment/SE/SE-Sync/'

    # ---------- Upload input frames: -------------------------------------------------------------------
    data = DataProvider()

    # ---------- Create graph (vertices and edges): -----------------------------------------------------
    graphMng = GraphManager(data)

    # ---------- Compute relative transformation for each edge in E: ------------------------------------
    graphMng.compute_relative_trans(sesync_path + 'data/')

    #  ---------- Run SE-Sync and get global SE transformations: -------------------------------------------
    run_sesync_in_matlab(sesync_path + 'MATLAB/examples')

    # ---------- Read the SE-Sync output: ---------------------------------------------------------------
    graphMng.read_sesync_output(sesync_path + "MATLAB/SE_Sync_output.txt")

    #  ---------- Transform the images and create a panorama (from frame0 to n_frame): ------------------
    n_frame = data.feed_size
    graphMng.transform_images_globally(n_frame, mypath + 'data/')

    #  ---------- Optimize the transformations using STN: -----------------------------------------------
    graphMng.prepare_data_for_STN(mypath + 'data/')  # only for debug


def run_sesync_in_matlab(sesync_path):
    eng = matlab.engine.start_matlab()
    eng.addpath(r'' + sesync_path, nargout=0)
    eng.main(nargout=0)
    eng.quit()

