# -*- coding: utf-8 -*-
"""
Updated on Thu Jun 14 11:08:01 2018

@author: wufei
"""


import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.GPU_ID = 0

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# father_path   = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
# grader_father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")


__C.IMAGE_TYPE = '.jpg'

__C.TRAIN_IMAGE_DIR = cfg.ROOT_DIR + '/trainimage'

__C.VAL_IMAGE_DIR = cfg.ROOT_DIR + '/dataset/validate'

__C.TEST_IMAGE_DIR = cfg.ROOT_DIR + '/testimage'


'''
we use 'HAED_' denotes the defined parameters for the traning of ship head classification network
'''
# the path where we put the samples and prototxt for ship head classification
__C.HEAD_DIR = cfg.ROOT_DIR + '/shiphead'

# the file which records the path of ship head samples and the label
__C.HEAD_DATA_TXT = cfg.HEAD_DIR + '/shipheadsample.txt'

__C.HEAD_PRETRAINED_MODEL = None

# the prototxt stores some parameters for traning
__C.HEAD_TRAIN_SOLVER = cfg.HEAD_DIR + '/solver.prototxt'

__C.HEAD_SAVED_MODEL_NAME = cfg.HEAD_DIR + '/shiphead8'

__C.HEAD_BATCH_SIZE = 256

__C.HEAD_TEST_MODEL = cfg.HEAD_DIR + '/test.caffemodel'

__C.HEAD_TEST_PROTOTXT = cfg.HEAD_DIR + '/deploy.prototxt'



'''
we use 'BODY_' denotes the defined parameters for ship localization
'''
# the path which stores the samples and prototxt for iterative bounding-box regression (ship localization)
__C.BODY_DIR = cfg.ROOT_DIR + '/shipbody'

__C.BODY_PRETRAINED_MODEL = None

__C.BODY_TRAIN_SOLVER = cfg.BODY_DIR + '/solver.prototxt'

__C.BODY_SAVED_MODEL_NAME = cfg.BODY_DIR + '/ship'

__C.BODY_BATCH_SIZE = 64

__C.BODY_TEST_MODEL = cfg.BODY_DIR + '/test.caffemodel'

__C.BODY_TEST_PROTOTXT = cfg.BODY_DIR + '/test.prototxt'



# Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
__C.NMS = 0.5
__C.USE_GPU_NMS = False

# threshold to determine ship or non-ship 
__C.DETECT_THRESH = 0.95

# the total number of images after augmentation
__C.IMAGE_TOATL = 2400

# the image size
__C.IMAGE_WIDTH = 1024
__C.IMAGE_HEIGHT = 768

# the size of ship head samples (40*40)
# here we use a half of the size
__C.HEAD_PATCH_RADIUS = int(40/2)

__C.BODY_PATCH_WIDTH = 200
__C.BODY_PATCH_HEIGHT = 200

__C.PROPOSAL_NUMBER = 50


# Reversed
__C.OFFSET = -1

# Pixel mean values 127.5 = 255.0/2.0
__C.IMAGE_MEANS = np.array([[[127.5, 127.5, 127.5]]])

# 0.00784314 = 2.0/255.0
__C.IMAGE_SCALE = np.array([[[0.00784314, 0.00784314, 0.00784314]]])

__C.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# save the weights file every 10000 iters
__C.SNAPSHOT_EPOH = 10000


'''
mean and standard deviation for x,y,w,h. The purpose is to make the regression targets having 0 mean and 1 variance
The corresponding mean (denoted as _MEAN) and standard deviation (denoted as _STD) would change if you restart generating new training samples.
We put the values to the file of __C.STD_MEAN when updating the samples.
Therefore, actually you do not need to replace the corresponding values below by yourself
'''
__C.CENTERX_STD  = 0.0683
__C.CENTERX_MEAN = -0.0014
__C.CENTERY_STD  = 0.0665
__C.CENTERY_MEAN = -8.0603e-04

__C.WIDTH_STD    = 0.3822
__C.WIDTH_MEAN   = -0.2902
__C.HEIGHT_STD   = 0.4365
__C.HEIGHT_MEAN  = -0.3215

# the file to store the new values of the above eight parameters if you restart generating new training samples.
__C.STD_MEAN = cfg.BODY_DIR + '/parameter.npy'


__C.IMAGE_LIST = range(1, 200)
# The numbers correspond to numbers of ships in each training image. We assign higher weight to image with more ships
__C.IMAGE_WEIGHT = [4,4,4,2,7,6,8,8,8,8,8,6,8,12,9,6,7,11,6,5,6,7,12,8,7,6,10,8,10,5,6,5,12,4,6,8,7,6,6,6, \
                    6,7,6,6,6,9,5,6,6,6,10,6,10,10,8,6,13,6,6,8,8,8,5,7,6,7,6,6,6,7,7,5,5,5,7,7,6,6,7,7,9, \
                    10,8,11,10,10,8,10,9,10,8,7,11,9,6,8,9,6,6,8,7,6,8,9,9,8,8,8,11,9,4,7,6,7,7,4,17,13,12, \
                    13,21,9,11,7,12,11,13,12,13,7,10,10,10,10,8,7,7,8,8,7,5,9,9,8,9,7,8,11,8,12,13,12,9,11,8, \
                    8,8,10,8,9,11,13,7,7,8,9,13,8,8,8,5,6,7,10,8,6,9,13,9,13,8,8,13,13,8,7,13,8,5,5,5,4,4,4,5,5,10,7,6,8]
__C.TABLE = [z for x,y in zip(cfg.IMAGE_LIST, cfg.IMAGE_WEIGHT) for z in [x] * y]  
