#!/usr/bin/env python

import _init_paths
import caffe
from detect.test import test_image
from detect.config import cfg
import numpy as np
import os.path as osp

if __name__ == '__main__':
   
    parameter = np.load(cfg.STD_MEAN)
    cfg.CENTERX_STD  = parameter[0]
    cfg.CENTERX_MEAN = parameter[1]
    cfg.CENTERY_STD  = parameter[2]
    cfg.CENTERY_MEAN = parameter[3]

    cfg.WIDTH_STD    = parameter[4]
    cfg.WIDTH_MEAN   = parameter[5]
    cfg.HEIGHT_STD   = parameter[6]
    cfg.HEIGHT_MEAN  = parameter[7]
    
    
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

#    image_source = osp.join(cfg.ROOT_DIR, 'testimage/1.jpg')
    cfg.TEST_PROTOTXT = osp.join(cfg.ROOT_DIR, 'test.prototxt')
    cfg.TEST_MODEL    = osp.join(cfg.ROOT_DIR, 'demo.caffemodel')
#    net = caffe.Net(cfg.TEST_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
#    test_image(net, image_source, vis=True, save=False)

    for index in range(1, 81):
        cfg.IMAGE_NUMBER = index
        image_source = osp.join(cfg.ROOT_DIR + '/testimage', str(index) + cfg.IMAGE_TYPE)
        net = caffe.Net(cfg.TEST_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
        if osp.exists(image_source):
            test_image(net, image_source, vis=False, save=True)
