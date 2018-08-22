# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:58:04 2018

@author: wufei
"""

import caffe
import numpy as np
import cv2
from detect.config import cfg

def get_samples_blob(selected_ind, label):

    data_blob = np.zeros((cfg.HEAD_BATCH_SIZE, cfg.HEAD_PATCH_RADIUS*2, cfg.HEAD_PATCH_RADIUS*2, 3), dtype=np.float32)
    label_blob = np.zeros((cfg.HEAD_BATCH_SIZE, 1), dtype=np.int16)

    ind = 0
    for i in selected_ind:
        img = cv2.imread(label[i]['image'])
        im = img.astype(np.float32, copy=False)
        im -= cfg.IMAGE_MEANS
        im *= cfg.IMAGE_SCALE 
        data_blob[ind, 0:im.shape[0], 0:im.shape[1], :] = im
        label_blob[ind, 0] = label[i]['label']
        ind += 1

    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    blobs = {'data': data_blob}
    blobs['label'] = label_blob

    return blobs

class RoIDataLayer(caffe.Layer):
    """data layer used for training."""
    def set_label(self, label, index_record):
        self._label = label

        # this parameter is not used
        #self._index_record = index_record

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        self._label = {}
        #self._index_record = {}
        self._name_to_top_map = {}

        # data blob: a batch of ship head samples
        top[0].reshape(cfg.HEAD_BATCH_SIZE, 3, cfg.HEAD_PATCH_RADIUS*2, cfg.HEAD_PATCH_RADIUS*2)
        self._name_to_top_map['data'] = 0

        # label blob
        top[1].reshape(cfg.HEAD_BATCH_SIZE, 1)
        self._name_to_top_map['label'] = 1

    def forward(self, bottom, top):
        # randomly get a bacth of samples
        selected_ind = np.random.choice(np.arange(len(self._label)), cfg.HEAD_BATCH_SIZE, False)

        # get the input data for the data layer in network
        blobs = get_samples_blob(selected_ind, self._label)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
