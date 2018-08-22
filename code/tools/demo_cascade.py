#!/usr/bin/env python

import _init_paths
import caffe
from detect.test_image_cascade import test_image,test_shiphead
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
    '''
    image_source = cfg.ROOT_DIR + '/testimage/1.jpg'

    step1_PROTOTXT = 'test_shiphead.prototxt'
    cfg.TEST_MODEL = '/shiphead/shiphead_2_iter_30000.caffemodel'    

    net1 = caffe.Net(step1_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
    rois = test_shiphead(net1, image_source)

#    step2_PROTOTXT = 'test_cascade_2.prototxt'
#    cfg.TEST_MODEL = 'demo_cascade_2.caffemodel'

#    net2 = caffe.Net(step2_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
#    test_image(net2, image_source, rois, vis=True, save=False)

    '''
    for index in range(1, 81):
        cfg.IMAGE_NUMBER = index
        image_source = osp.join(cfg.ROOT_DIR + '/testimage', str(index)+cfg.IMAGE_TYPE)

        if osp.exists(image_source):
            step1_PROTOTXT = osp.join(cfg.ROOT_DIR, 'test_shiphead.prototxt')
            cfg.TEST_MODEL = osp.join(cfg.ROOT_DIR, 'shiphead/shiphead_2_iter_30000.caffemodel')

            net1 = caffe.Net(step1_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
            rois = test_shiphead(net1, image_source)

            step2_PROTOTXT = osp.join(cfg.ROOT_DIR, 'test_cascade_2.prototxt')
            cfg.TEST_MODEL = osp.join(cfg.ROOT_DIR, 'demo_cascade_2.caffemodel')

            net2 = caffe.Net(step2_PROTOTXT, cfg.TEST_MODEL, caffe.TEST)
            test_image(net2, image_source, rois, vis=False, save=True)
