# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:44:58 2018

@author: wufei
"""

import caffe
import numpy as np
import shutil  
import pandas as pd
from detect.config import cfg


def filter_shiphead_classification_result(datatxt, model_def, model_weights):

    # the path to put the updated datatxt
    datatxt_new = cfg.HEAD_DIR + '/shipheadnew.txt'
    
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    mu = np.array([1.0, 1.0, 1.0])
    
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # move image channels to outermost dimension
    transformer.set_transpose('data', (2,0,1))
    # rescale from [0, 1] to [0, 2]
    transformer.set_raw_scale('data', 2)
    # subtract the dataset-mean value in each channel
    transformer.set_mean('data', mu)
    # swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2,1,0))
    
    # set the size of the net input
    net.blobs['data'].reshape(1, 3, cfg.HEAD_PATCH_RADIUS*2, cfg.HEAD_PATCH_RADIUS*2)

    shutil.copyfile(datatxt, datatxt_new)

    with open(datatxt_new, "a") as f:
        f.write('\n')

    # read the label of each sample
    data_list = pd.read_table(datatxt, header=None, sep='[ ]', engine='python')

    for i in range(0, len(data_list)):
        if i%2000 == 0:
            print ('Processing ' + str(i))

        imagePath = str(data_list.iat[i,0])

        image = caffe.io.load_image(imagePath)
        transformed_image = transformer.preprocess('data', image)

        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        # the output probability vector for the input image
        output_prob = output['prob'][0]

        # add the incorrectly-classified samples to the new datatxt
        if data_list.iat[i,1] != output_prob.argmax():
            with open(datatxt_new, "a") as f:
                record = str(imagePath) + ' ' + str(data_list.iat[i,1]) + '\n'
                f.write(record)

    return datatxt_new