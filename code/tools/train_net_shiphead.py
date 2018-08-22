#!/usr/bin/env python

import _init_paths
import caffe
import copy
import pandas as pd
from detect.config import cfg
from detect.shiphead_train import train_net
from detect.filter_shiphead_classification_result import filter_shiphead_classification_result
from detect.shiphead_sample import generate_shiphead_samples


def extract_data(filename):
    data = dict()
    datadb = list()
    indexdb = list()
    index = 0

    data_list = pd.read_table(filename, header=None, sep='[ ]', engine='python')

    for i in range(0, len(data_list)):
        data['image'] = str(data_list.iat[i, 0])
        data['label'] = int(data_list.iat[i, 1])
        datadb.append(copy.deepcopy(data))
        indexdb.append(int(index))
        index = index + 1

    return datadb, indexdb

if __name__ == '__main__':

    '''
    The following parameters are defined and initialized in config.py
        cfg.HEAD_TRAIN_SOLVER : cfg.HEAD_DIR + '/solver.prototxt'
        cfg.TRAIN_IMAGE_DIR : cfg.ROOT_DIR + '/trainimage'
        cfg.HEAD_DATA_TXT : cfg.HEAD_DIR + '/shipheadtrainsample.txt'
        cfg.HEAD_PRETRAINED_MODEL : None
        cfg.HEAD_SAVED_MODEL_NAME : cfg.HEAD_DIR + '/shiphead'
        cfg.HEAD_TEST_PROTOTXT : cfg.HEAD_DIR + '/deploy.prototxt
        cfg.HEAD_TEST_MODEL = cfg.HEAD_DIR + '/test.caffemodel'
    '''
    # iterations in the first stage
    iter_step1 = 30000
    # iterations in the second stage
    iter_step2 = 30000

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    # generate the training samples for ship head classification
    generate_shiphead_samples(cfg.TRAIN_IMAGE_DIR, cfg.HEAD_DATA_TXT, cfg.HEAD_DIR)

    # the first training step for ship head classification
    data_record, index_record = extract_data(cfg.HEAD_DATA_TXT)
    train_net(cfg.HEAD_TRAIN_SOLVER, data_record, index_record, pretrained_model = cfg.HEAD_PRETRAINED_MODEL, max_iters = iter_step1)

    # update the training samples for ship head classification
    cfg.HEAD_TEST_MODEL = cfg.HEAD_SAVED_MODEL_NAME + '_iter_{:d}'.format(iter_step1) + '.caffemodel'
    cfg.HEAD_DATA_TXT = filter_shiphead_classification_result(cfg.HEAD_DATA_TXT, cfg.HEAD_TEST_PROTOTXT, cfg.HEAD_TEST_MODEL)

    # the second training step for ship head classification
    cfg.HEAD_TRAIN_SOLVER = cfg.HEAD_DIR + '/solver_2.prototxt'
    cfg.HEAD_SAVED_MODEL_NAME = cfg.HEAD_DIR + '/shiphead_2'
    cfg.HEAD_PRETRAINED_MODEL = cfg.HEAD_TEST_MODEL
    data_record_new, index_record_new = extract_data(cfg.HEAD_DATA_TXT)
    train_net(cfg.HEAD_TRAIN_SOLVER, data_record_new, index_record_new, pretrained_model = cfg.HEAD_PRETRAINED_MODEL, max_iters = iter_step2)    