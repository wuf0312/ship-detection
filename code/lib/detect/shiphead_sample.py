# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:53:04 2018

@author: wufei
"""

import os
import os.path as osp
import numpy as np
import random
import cv2
from detect.config import cfg


def generate_shiphead_samples(train_image_dir, txt_dir, head_dir):

    '''
    Generate ship head samples for ship head classification
    
    Parameter
        train_image_dir : the path containing the training images
        txt_dir         : the full path of txt file which records the labels for ship head samples
        head_dir        : we put this txt file in this folder
    '''
    # we put the ship head samples in this folder
    sample_dir = head_dir + '/train'
    if not osp.exists(sample_dir):
        os.makedirs(sample_dir)
        startNumber = 0
        if osp.exists(txt_dir):
            os.remove(txt_dir)
        startNumber = generate_shiphead_samples_positive(train_image_dir, txt_dir, sample_dir, startNumber)
        generate_shiphead_samples_negative_normal(train_image_dir, txt_dir, sample_dir, startNumber)

    else:
        print('The path containing ship head samples already exists. If you want to generate new samples again, please delete the folder containing the samplse, and run this function again.')
        print('The path is ' + sample_dir)

def generate_shiphead_samples_positive(image_dir, txt_dir, sample_dir, startNumber):

    '''
    Generate positive samples for ship head classification
    Here we divide ship heads with different directions into 8 bins
    
    Parameter
        image_dir       : the path containing the training images
        txt_dir         : the full path of txt file which records the labels for ship head samples
        sample_dir      : we put the ship head samples in this folder
        startNumber     : the start number of samples (the name of samples)
    '''

    radius = cfg.HEAD_PATCH_RADIUS
    offset = int(radius*0.6)

    for index in range(1, cfg.IMAGE_TOATL + 1):
        imagePath = osp.join(image_dir, str(index) + cfg.IMAGE_TYPE)
        if index <= cfg.IMAGE_TOATL/2:
            recordPath = osp.join(image_dir, str(index) +'.txt')
        else:
            recordPath = osp.join(image_dir, str(index-cfg.IMAGE_TOATL/2) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):
            print(imagePath)
            data1 = []
            data2 = []
            data3 = []
            data4 = []
            f = open(recordPath, 'r')
            for line in f.readlines():
                line = line.strip()
                if not len(line) or line.startswith('#'):
                    continue

                data1_tmp ,data2_tmp, data3_tmp, data4_tmp, data5_tmp, data6_tmp, data7_tmp, \
                    data101_tmp, data102_tmp, data103_tmp, data104_tmp = [float(i) for i in line.split()]
                # in the txt file which stores the position of ground-truth bounding box
                # the 5th data of each line denotes the label of object, and 2 denotes the ships which we want to detect
                if int(data5_tmp) == 2:
                    data1.append(data1_tmp + cfg.OFFSET)
                    data2.append(data2_tmp + cfg.OFFSET)
                    data3.append(data3_tmp + cfg.OFFSET)
                    data4.append(data4_tmp + cfg.OFFSET)

            data1 = np.array(data1, dtype = int)
            data2 = np.array(data2, dtype = int)
            data3 = np.array(data3, dtype = int)
            data4 = np.array(data4, dtype = int)

            shipNumber = len(data1)
            im = cv2.imread(imagePath)

            im1 = cv2.copyMakeBorder(im, 2*radius, 2*radius, 2*radius, 2*radius, cv2.BORDER_REPLICATE)

            height, width, channels = im1.shape

            data1[:] = data1[:] + 2*radius
            data2[:] = data2[:] + 2*radius
            data3[:] = data3[:] + 2*radius
            data4[:] = data4[:] + 2*radius

            for j in range(0, shipNumber):
                if(data1[j]>radius*3 and data2[j]>radius*3 and data1[j]<width-radius*3 and data2[j]<height-radius*3):
                    imOriginPatch = im1[data2[j]-radius*3:data2[j]+radius*3, data1[j]-radius*3:data1[j]+radius*3, :]
                    theta = np.arctan2(data3[j] - data1[j], data4[j] - data2[j]) * 180 / np.pi
                    for n in range(0, 360, 45):
                        th = -theta + 90 - n + 45 * (random.uniform(-0.5, 0.5))
                        M = cv2.getRotationMatrix2D((int(imOriginPatch.shape[0]/2.), int(imOriginPatch.shape[1]/2.)), th, 1.0)
                        imPatch = cv2.warpAffine(imOriginPatch, M, (imOriginPatch.shape[0], imOriginPatch.shape[1]))
                        center = int(imPatch.shape[0]/2.)

                        for m in range(0, 1):
                            if n == 0:
                                neww = center + random.randint(0, offset*2)
                                newh = center + random.randint(-offset, offset)
                            elif n == 45:
                                neww = center + random.randint(0, offset*2)
                                newh = center + random.randint(0, offset*2)
                            elif n == 90:
                                neww = center + random.randint(-offset, offset)
                                newh = center + random.randint(0, offset*2)
                            elif n == 135:
                                neww = center + random.randint(-offset*2, 0)
                                newh = center + random.randint(0, offset*2)
                            elif n == 180:
                                neww = center + random.randint(-offset*2, 0)
                                newh = center + random.randint(-offset, offset)
                            elif n == 225:
                                neww = center + random.randint(-offset*2, 0)
                                newh = center + random.randint(-offset*2, 0)
                            elif n == 270:
                                neww = center + random.randint(-offset, offset)
                                newh = center + random.randint(-offset*2, 0)
                            elif n == 315:
                                neww = center + random.randint(0, offset*2)
                                newh = center + random.randint(-offset*2, 0)
                            else:
                                continue
                            try:
                                imageShiphead = imPatch[newh-radius:newh+radius, neww-radius:neww+radius, :]
                                startNumber  = startNumber + 1
                                shipheadPath = osp.join(sample_dir, str(startNumber + int(n/45 + 1)*1000000) + cfg.IMAGE_TYPE)
                                cv2.imwrite(shipheadPath, imageShiphead)
                                with open(txt_dir, "a") as f:
                                   record = str(shipheadPath) + ' ' + str(int(n/45 + 1)) + '\n'
                                   f.write(record)
                            except:
                                continue
    return startNumber



def generate_shiphead_samples_negative_normal(image_dir, txt_dir, sample_dir, startNumber):

    '''
    Generate negative samples for ship head classification
    the samples are randomly generated from the background 
    
    Parameter
        image_dir       : the path containing the training images
        txt_dir         : the full path of txt file which records the labels for ship head samples
        sample_dir      : we put the ship head samples in this folder
        startNumber     : the start number of samples (the name of samples)
    '''

    radius = cfg.HEAD_PATCH_RADIUS
    offset = int(radius*1.0)
    # we ranomly abandon some samples that are with low mean and std,
    # therefore the number of samples from each image is lower than (samplePerimage)
    samplePerimage = 35

    for index in range(1, cfg.IMAGE_TOATL + 1):
        imagePath  = osp.join(image_dir, str(index) + cfg.IMAGE_TYPE)
        if index < cfg.IMAGE_TOATL/2:
            recordPath = osp.join(image_dir, str(index) + '.txt')
        else:
            recordPath = osp.join(image_dir, str(index - cfg.IMAGE_TOATL/2) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):
            print(imagePath)
            data1 = []
            data2 = []
            data3 = []
            data4 = []
            f = open(recordPath, 'r')
            for line in f.readlines():
                line = line.strip()
                if not len(line) or line.startswith('#'):
                    continue 

                data1_tmp ,data2_tmp, data3_tmp, data4_tmp, data5_tmp, data6_tmp, data7_tmp, \
                    data101_tmp, data102_tmp, data103_tmp, data104_tmp = [float(i) for i in line.split()]
                if int(data5_tmp) == 2:  
                    data1.append(data1_tmp + cfg.OFFSET)
                    data2.append(data2_tmp + cfg.OFFSET)
                    data3.append(data3_tmp + cfg.OFFSET)
                    data4.append(data4_tmp + cfg.OFFSET)

            shipNumber = len(data1)
            data1 = np.array(data1, dtype = int)
            data2 = np.array(data2, dtype = int)
            data3 = np.array(data3, dtype = int)
            data4 = np.array(data4, dtype = int)

            im = cv2.imread(imagePath)
            height, width, channels = im.shape

            newx1 = np.zeros((shipNumber), dtype=int)
            newy1 = np.zeros((shipNumber), dtype=int)
            newx2 = np.zeros((shipNumber), dtype=int)
            newy2 = np.zeros((shipNumber), dtype=int)

            for j in range(0, shipNumber):
                theta = np.arctan2(data3[j] - data1[j],  data4[j] - data2[j]) * 180 / np.pi
                if((theta>=180+157.5 and theta<=360) or (theta>=0 and theta<22.5)):
                    newx1[j] = data1[j] - offset*0.8
                    newy1[j] = data2[j] - offset*1.2
                    newx2[j] = data1[j] + offset*0.8
                    newy2[j] = data2[j] + offset*0.4
                elif(theta>=22.5 and theta<67.5):
                    newx1[j] = data1[j] - offset*1.2
                    newy1[j] = data2[j] - offset*1.2
                    newx2[j] = data1[j] + offset*0.4
                    newy2[j] = data2[j] + offset*0.4
                elif(theta>=67.5 and theta<112.5):
                    newx1[j] = data1[j] - offset*1.2
                    newy1[j] = data2[j] - offset*0.8
                    newx2[j] = data1[j] + offset*0.4
                    newy2[j] = data2[j] + offset*0.8
                elif(theta>=112.5 and theta<157.5):
                    newx1[j] = data1[j] - offset*1.2
                    newy1[j] = data2[j] - offset*0.4
                    newx2[j] = data1[j] + offset*0.4
                    newy2[j] = data2[j] + offset*1.2
                elif(theta>=180-22.5 and theta<180+22.5):
                    newx1[j] = data1[j] - offset*0.8
                    newy1[j] = data2[j] - offset*0.4
                    newx2[j] = data1[j] + offset*0.8
                    newy2[j] = data2[j] + offset*1.2
                elif(theta>=180+22.5 and theta<180+67.5):
                    newx1[j] = data1[j] - offset*0.4
                    newy1[j] = data2[j] - offset*0.4
                    newx2[j] = data1[j] + offset*1.2
                    newy2[j] = data2[j] + offset*1.2
                elif(theta>=180+67.5 and theta<180+112.5):
                    newx1[j] = data1[j] - offset*0.4
                    newy1[j] = data2[j] - offset*0.8
                    newx2[j] = data1[j] + offset*1.2
                    newy2[j] = data2[j] + offset*0.8
                elif(theta>=180+112.5 and theta<180+157.5):
                    newx1[j] = data1[j] - offset*0.4
                    newy1[j] = data2[j] - offset*1.2
                    newx2[j] = data1[j] + offset*1.2
                    newy2[j] = data2[j] + offset*0.4
                else:
                    continue

            number = 0
            while number < samplePerimage:
                neww = random.randint(radius*2, width-radius*2)
                newh = random.randint(radius*2, height-radius*2)
                savedata = 1
                for k in range(0, shipNumber):
                    if(savedata == 1 and neww+radius>newx1[k] and neww-radius<newx2[k] and newh+radius>newy1[k] and newh-radius<newy2[k]):
                        savedata = 0
                if savedata == 1:
                    number = number + 1
                    try:
                        image = im[newh-radius:newh+radius, neww-radius:neww+radius, :]
                        # we ranomly abandon some samples that are with low mean and std.
                        # they are mostly samples for sea area
                        if(np.mean(image)>25 and np.std(image)>16):
                            startNumber = startNumber + 1
                            nonshipheadPath = osp.join(sample_dir, str(startNumber) + cfg.IMAGE_TYPE)                          
                            cv2.imwrite(nonshipheadPath, image)
                            with open(txt_dir, "a") as f:
                                record = str(nonshipheadPath) + ' ' + str(0) + '\n'
                                f.write(record)
                        elif random.randint(1, 2) > 1:
                            startNumber = startNumber + 1
                            nonshipheadPath = osp.join(sample_dir, str(startNumber) + cfg.IMAGE_TYPE)
                            cv2.imwrite(nonshipheadPath, image)
                            with open(txt_dir, "a") as f:
                                record = str(nonshipheadPath) + ' ' + str(0) + '\n'
                                f.write(record)
                        else:
                            number = number - 1                            
                    except:
                        number = number - 1
                        continue
    return startNumber