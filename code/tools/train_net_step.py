#!/usr/bin/env python

import _init_paths
import caffe
import numpy as np
import copy
import os.path as osp
from detect.train import train_net
from detect.shipbody_RoIreg import generate_shipbody_sample, compute_regression_and_classification_target
from detect.config import cfg


def extract_data(filename):
    data = dict()
    roidb = list()
    indexdb = list()

    print(filename)
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue

        num, index, roitx, roity, roidx, roidy, bboxx, bboxy, bboxw, bboxh, label, weight = [float(i) for i in line.split()]
        data['roi'] = np.array([roitx, roity, roidx, roidy])
        data['bbox_target'] = np.array([bboxx, bboxy, bboxw, bboxh])
        data['label'] = int(label)
        data['weight'] = weight
        data['image'] = osp.abspath(osp.join(cfg.TRAIN_IMAGE_DIR, str(int(index)) + cfg.IMAGE_TYPE))
        data['imageindex'] = int(index)
        roidb.append(copy.deepcopy(data))
        indexdb.append(int(index))

    return sorted(roidb, key=lambda x: x['imageindex'], reverse=False), sorted(indexdb, key=lambda x: x, reverse=False)

if __name__ == '__main__':

    '''
    The followings are defined in config.py
        cfg.TRAIN_IMAGE_DIR   = cfg.ROOT_DIR + '/trainimage'
        cfg.BODY_TRAIN_SOLVER = cfg.BODY_DIR + '/solver_ship.prototxt'
        cfg.BODY_SAVED_MODEL_NAME = cfg.BODY_DIR + '/ship'
        cfg.BODY_PRETRAINED_MODEL = None
    '''

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

#    txt_path = osp.join(cfg.BODY_DIR, 'shipbody.txt')
    txt_path = generate_shipbody_sample(cfg.TRAIN_IMAGE_DIR)
    txt_target_path = compute_regression_and_classification_target(txt_path)

    parameter = np.load(cfg.STD_MEAN)
    cfg.CENTERX_STD  = parameter[0]
    cfg.CENTERX_MEAN = parameter[1]
    cfg.CENTERY_STD  = parameter[2]
    cfg.CENTERY_MEAN = parameter[3]
    cfg.WIDTH_STD    = parameter[4]
    cfg.WIDTH_MEAN   = parameter[5]
    cfg.HEIGHT_STD   = parameter[6]
    cfg.HEIGHT_MEAN  = parameter[7]

    data_record, index_record = extract_data(txt_target_path)

    # if you just want to fine-tune the netwrok, 
    # you can load the given model of "/shipbody/ship_2_iter_100000.caffemodel" (use the following code). 
    #cfg.BODY_TRAIN_SOLVER = cfg.BODY_DIR + '/solver_2.prototxt'
    #cfg.BODY_SAVED_MODEL_NAME = cfg.BODY_DIR + '/ship_2'
    #cfg.BODY_PRETRAINED_MODEL = cfg.BODY_DIR + '/ship_2_iter_100000.caffemodel'
    

    # if you want to train it with a fresh start, you can use the following code.
    iters_step1 = 50000
    train_net(cfg.BODY_TRAIN_SOLVER, data_record, index_record, pretrained_model = cfg.BODY_PRETRAINED_MODEL, max_iters=iters_step1)
    cfg.BODY_PRETRAINED_MODEL = cfg.BODY_DIR + '/' + 'ship_iter_{:d}'.format(iters_step1) + '.caffemodel'
    cfg.BODY_TRAIN_SOLVER = cfg.BODY_DIR + '/solver_2.prototxt'
    cfg.BODY_SAVED_MODEL_NAME = cfg.BODY_DIR + '/ship_2'


    iters_step2 = 100000
    train_net(cfg.BODY_TRAIN_SOLVER, data_record, index_record, pretrained_model = cfg.BODY_PRETRAINED_MODEL, max_iters=iters_step2)