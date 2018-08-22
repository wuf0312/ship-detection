# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:50:34 2018

@author: wufei
"""

import sys, os
import os.path as osp
import numpy as np
import random
import cv2
from detect.config import cfg

debug = False


def generate_shipbody_sample(imageDir):
    '''
    Generate training samples for the ship localization step
    
    Parameter
        imageDir : the path of the training images (after augmentation)
    Return
        recordTxt: the txt file which stores the positions of the samples
    '''

    # the path where we put the training samples
    samplePath = cfg.BODY_DIR;

    if not osp.exists(samplePath):
         os.mkdir(samplePath)
    
    #the txt file which stores the positions of the samples
    recordTxt = osp.join(samplePath, 'shipbody.txt')

    if osp.exists(recordTxt):
        print('The generated ship samples already exists. If you want to generate new samples again, please delete the txt file, and run this function again.')
        print('The txt file is ' + recordTxt)
        return recordTxt
   
    totalNumber = 0
    radius = cfg.BODY_PATCH_WIDTH/2
    # we sample k_threshold samples around each ship target
    k_threshold = 32

    #data1, data2, data3, data4, data5, data6, data7, data101, data102, data103  data104
    #起点x  起点y  终点x  终点y  类别   中点x  中点y  框左上x  框左上y  框右下x  框右下y
    for index in range(1, cfg.IMAGE_TOATL + 1):
        imagePath = osp.join(imageDir, str(index) + cfg.IMAGE_TYPE)
        if(index>1200 and index<2401):
            recordPath = osp.join(imageDir, str(index-1200) + '.txt')
        else:
            recordPath = osp.join(imageDir, str(index) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):
            print (imagePath)
            '''
            data1, data2: the position of ship head
            data3, data4: the position of stern
            data5: label
            data6, data7: the position of midpoint
            data101, data102: the position of up-left point of ground truth
            data103  data104: the position of down-right point of ground truth
            '''
            data1 = []
            data2 = []
            data3 = []
            data4 = []
            data5 = []
            data6 = []
            data7 = []
            data101 = []
            data102 = []
            data103 = []
            data104 = []

            f = open(recordPath, 'r')
            for line in f.readlines():
                line = line.strip()
                if not len(line) or line.startswith('#'):
                    continue

                data1_tmp ,data2_tmp, data3_tmp, data4_tmp, data5_tmp, data6_tmp, data7_tmp, \
                    data101_tmp, data102_tmp, data103_tmp, data104_tmp = [float(i) for i in line.split()]
                # the 5th data in each line is the label for target, and when label==2, the target is the ship to be detected
                if int(data5_tmp) == 2:
                    data1.append(data1_tmp + cfg.OFFSET)
                    data2.append(data2_tmp + cfg.OFFSET)
                    data3.append(data3_tmp + cfg.OFFSET)
                    data4.append(data4_tmp + cfg.OFFSET)
                    data5.append(data5_tmp)
                    data6.append((data101_tmp+data103_tmp)/2. + cfg.OFFSET)
                    data7.append((data102_tmp+data104_tmp)/2. + cfg.OFFSET)
                    data101.append( min(data101_tmp, data103_tmp) + cfg.OFFSET)
                    data102.append( min(data102_tmp, data104_tmp) + cfg.OFFSET)
                    data103.append( max(data101_tmp, data103_tmp) + cfg.OFFSET)
                    data104.append( max(data102_tmp, data104_tmp) + cfg.OFFSET)

            data_temp = np.zeros(len(data1))
            data_temp[:] = cfg.IMAGE_WIDTH ** 2 + cfg.IMAGE_HEIGHT ** 2
#            if not np.where(np.array(data5)==2):
#                continue

            for j in range(0, len(data1)):
                k = 0;
                itera = 0;
                while k < k_threshold:
                    if itera > k_threshold*10:
                        break
                    else:
                        itera = itera + 1

                        newkx = random.randint(data101[j], data103[j])
                        newky = random.randint(data102[j], data104[j])
                        x1 = max(0, newkx-radius);
                        x2 = min(newkx+radius-1, cfg.IMAGE_WIDTH-1);
                        y1 = max(0, newky-radius);
                        y2 = min(newky+radius-1, cfg.IMAGE_HEIGHT-1);
                        newkx = int((x1+x2)/2.);
                        newky = int((y1+y2)/2.);

                        for m in range(0, len(data1)):
                            data_temp[m] = (newkx-data6[m]) ** 2 + (newky-data7[m]) ** 2;

                        I = np.argmin(data_temp);

                        interminx = max(x1, data101[I])
                        interminy = max(y1, data102[I])
                        intermaxx = min(x2, data103[I])
                        intermaxy = min(y2, data104[I])

                        ratio = (intermaxx-interminx)*(intermaxy-interminy)/abs((data104[I]-data102[I])*(data103[I]-data101[I]));
                        if(((x1 < data1[I] and x2 > data1[I] and y1 < data2[I] and y2 > data2[I]) or
                            (x1 < data3[I] and x2 > data3[I] and y1 < data4[I] and y2 > data4[I])) and ratio>0.5):

                            totalNumber = totalNumber + 1
                            k = k + 1
                            if debug:
                                im = cv2.imread(imagePath)
                                image = im[y1:y2+1, x1:x2+1, :]
                                cv2.imwrite(osp.join(samplePath, str(totalNumber) + cfg.IMAGE_TYPE), image)
                            '''
                            the record contain the folowings:
                            reserved (0), from the index_th traning image, the bounding box of proposal (the x and y of Upper left and Down Right point),
                            the bounding box of its ground truth (also the x and y of Upper left and Down Right point), label 
                            '''
                            with open(recordTxt, "a") as f:
                                record = str(0) + ' ' + str(index) + ' ' +  str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' +\
                                    str(int(data101[j])) + ' ' + str(int(data102[j])) + ' ' + \
                                    str(int(data103[j])) + ' ' + str(int(data104[j])) + ' ' + str(1)  + '\n'
                                f.write(record)
    return recordTxt

def compute_regression_and_classification_target(srcTxt):

    '''
    Compute the regression target and label
    
    Parameter
        srcTxt: the txt file which stores the positions of the samples
    Return
        step2txt: the txt file which stores the regression targets and labels of the samples
    '''

    step2txt = osp.join(cfg.BODY_DIR, 'shipbodystep2.txt')

    if osp.exists(step2txt):
        print('The file containing regression and classification traning targets already exists. If you want to compute them again, please delete the txt file, and run this function again.')
        print('The txt file is ' + step2txt)
        return step2txt

    step = 2

    '''
    data1 : in the ith traning image
    patchx1, patchy1 : position of up-left point (proposal)
    patchx2, patchy2 : position of down-right point (proposal)
    gtx1, gty1 : position of up-left point (ground truth)
    gtx2, gty2 : position of down-right point (ground truth)
    '''  
    data1 = []
    patchx1 = []
    patchy1 = []
    patchx2 = []
    patchy2 = []
    gtx1 = []
    gty1 = []
    gtx2 = []
    gty2 = []

    f = open(srcTxt, 'r')
    for line in f.readlines():
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue 

        data0_tmp, data1_tmp, patchx1_tmp, patchy1_tmp, patchx2_tmp, patchy2_tmp, \
            gtx1_tmp, gty1_tmp, gtx2_tmp, gty2_tmp, label_tmp = [float(i) for i in line.split()]
        data1.append(data1_tmp)
        patchx1.append(patchx1_tmp)
        patchy1.append(patchy1_tmp)
        patchx2.append(patchx2_tmp)
        patchy2.append(patchy2_tmp)
        gtx1.append(gtx1_tmp)
        gtx2.append(gtx2_tmp)
        gty1.append(gty1_tmp)
        gty2.append(gty2_tmp)

    data1 = np.array(data1)
    patchx1 = np.array(patchx1)
    patchy1 = np.array(patchy1)
    patchx2 = np.array(patchx2)
    patchy2 = np.array(patchy2)
    gtx1 = np.array(gtx1)
    gty1 = np.array(gty1)
    gtx2 = np.array(gtx2)
    gty2 = np.array(gty2)

    length = len(data1)

    x1step1 = np.zeros(length)
    y1step1 = np.zeros(length)
    x2step1 = np.zeros(length)
    y2step1 = np.zeros(length)
    x1step2 = np.zeros(length)
    y1step2 = np.zeros(length)
    x2step2 = np.zeros(length)
    y2step2 = np.zeros(length)

    # Ground truth x1 y1 x2 y2,   in the 1st step
    x1step1[:] = gtx1[:]*1./step + patchx1[:]*1./step
    y1step1[:] = gty1[:]*1./step + patchy1[:]*1./step
    x2step1[:] = gtx2[:]*1./step + patchx2[:]*1./step
    y2step1[:] = gty2[:]*1./step + patchy2[:]*1./step
    # Ground truth x1 y1 x2 y2,   in the 2nd step
    x1step2[:] = gtx1[:]
    y1step2[:] = gty1[:]
    x2step2[:] = gtx2[:]
    y2step2[:] = gty2[:]


    midxstep0 = np.zeros(length)
    midystep0 = np.zeros(length)
    widstep0 = np.zeros(length)
    heistep0 = np.zeros(length)
    midxstep1 = np.zeros(length)
    midystep1 = np.zeros(length)
    widstep1 = np.zeros(length)
    heistep1 = np.zeros(length)
    midxstep2 = np.zeros(length)
    midystep2 = np.zeros(length)
    widstep2 = np.zeros(length)
    heistep2 = np.zeros(length)

    # proposal midx midy width height,  in the 1st step
    midxstep0[:] = (patchx1[:] + patchx2[:])/2.
    midystep0[:] = (patchy1[:] + patchy2[:])/2.
    widstep0[:]  = patchx2[:] - patchx1[:]
    heistep0[:]  = patchy2[:] - patchy1[:]
    # Ground truth midx midy width height,  in the 1st step /i.e. proposal in the 2nd step
    midxstep1[:] = (x1step1[:] + x2step1[:])/2.
    midystep1[:] = (y1step1[:] + y2step1[:])/2.
    widstep1[:]  = x2step1[:] - x1step1[:]
    heistep1[:]  = y2step1[:] - y1step1[:]
    # Ground truth midx midy width height,  in the 2nd step
    midxstep2[:] = (x1step2[:] + x2step2[:])/2.
    midystep2[:] = (y1step2[:] + y2step2[:])/2.
    widstep2[:]  = x2step2[:] - x1step2[:]
    heistep2[:]  = y2step2[:] - y1step2[:]

    # N1 :the 1st iteration bounding-box regression target
    N1 = np.zeros((len(data1), 4), dtype = np.float32)
    NN1 = np.zeros((len(data1), 4), dtype = np.float32)

    N1[:,0] = (midxstep1[:] - midxstep0[:])/widstep0[:]
    N1[:,1] = (midystep1[:] - midystep0[:])/heistep0[:]
    N1[:,2] = np.log(widstep1[:]/widstep0[:])
    N1[:,3] = np.log(heistep1[:]/heistep0[:])
    
    # N2 :the 2nd iteration bounding-box regression target
    N2 = np.zeros((len(data1), 4), dtype = np.float32)
    NN2 = np.zeros((len(data1), 4), dtype = np.float32)

    N2[:,0] = (midxstep2[:] - midxstep1[:])/widstep1[:]
    N2[:,1] = (midystep2[:] - midystep1[:])/heistep1[:]
    N2[:,2] = np.log(widstep2[:]/widstep1[:])
    N2[:,3] = np.log(heistep2[:]/heistep1[:])

    # N3 :the newly-added bounding-box regression target(from groundtruth to groundtruth)
    N3 = np.zeros((len(data1), 4), dtype = np.float32)
    NN3 = np.zeros((len(data1), 4), dtype = np.float32)

    N3[:,0] = (midxstep2[:] - midxstep2[:])/widstep2[:]
    N3[:,1] = (midystep2[:] - midystep2[:])/heistep2[:]
    N3[:,2] = np.log(widstep2[:]/widstep2[:])
    N3[:,3] = np.log(heistep2[:]/heistep2[:])


    XYZ = np.concatenate((N1, N2, N3), axis=0)

    std2data1 = np.std(XYZ[:,0])
    meandata1 = np.mean(XYZ[:,0])
    std2data2 = np.std(XYZ[:,1])
    meandata2 = np.mean(XYZ[:,1])
    std2data3 = np.std(XYZ[:,2])
    meandata3 = np.mean(XYZ[:,2])
    std2data4 = np.std(XYZ[:,3])
    meandata4 = np.mean(XYZ[:,3])

    cfg.CENTERX_STD  = std2data1
    cfg.CENTERX_MEAN = meandata1
    cfg.CENTERY_STD  = std2data2
    cfg.CENTERY_MEAN = meandata2

    cfg.WIDTH_STD    = std2data3
    cfg.WIDTH_MEAN   = meandata3
    cfg.HEIGHT_STD   = std2data4
    cfg.HEIGHT_MEAN  = meandata4

    parameter = [std2data1, meandata1, std2data2, meandata2, std2data3, meandata3, std2data4, meandata4]
    np.save(cfg.STD_MEAN, parameter)
    
    NN1[:,0] = (N1[:,0] - meandata1)/std2data1
    NN1[:,1] = (N1[:,1] - meandata2)/std2data2
    NN1[:,2] = (N1[:,2] - meandata3)/std2data3
    NN1[:,3] = (N1[:,3] - meandata4)/std2data4

    NN2[:,0] = (N2[:,0] - meandata1)/std2data1
    NN2[:,1] = (N2[:,1] - meandata2)/std2data2
    NN2[:,2] = (N2[:,2] - meandata3)/std2data3
    NN2[:,3] = (N2[:,3] - meandata4)/std2data4

    NN3[:,0] = (N3[:,0] - meandata1)/std2data1
    NN3[:,1] = (N3[:,1] - meandata2)/std2data2
    NN3[:,2] = (N3[:,2] - meandata3)/std2data3
    NN3[:,3] = (N3[:,3] - meandata4)/std2data4

    '''
    the record contain the followings:
    reserved (0), from the index_th traning image, the bounding box of proposal (the x and y of Upper left and Down Right point),
    the regression target, label, weight(reserved)
    '''
    for j in range(0, length):
        ix1 = max(patchx1[j], x1step2[j])
        iy1 = max(patchy1[j], y1step2[j])
        ix2 = min(patchx2[j], x2step2[j])
        iy2 = min(patchy2[j], y2step2[j])
        IoU = (iy2-iy1+1)*(ix2-ix1+1)/((y2step2[j]-y1step2[j]+1)*(x2step2[j]-x1step2[j]+1)+(patchy2[j]-patchy1[j]+1)*(patchx2[j]-patchx1[j]+1)-(iy2-iy1+1)*(ix2-ix1+1))
        if IoU >= 0.7:
            with open(step2txt, "a") as f:
                record = str(0) + ' ' + str(int(data1[j])) + ' ' + \
                    str(int(patchx1[j])) + ' ' + str(int(patchy1[j])) + ' ' +  str(int(patchx2[j])) + ' ' + str(int(patchy2[j])) + ' ' + \
                    str(NN1[j,0]) + ' ' + str(NN1[j,1]) + ' ' + str(NN1[j,2]) + ' ' + str(NN1[j,3]) + ' ' + str(1) + ' ' + str(1) + '\n'
                f.write(record)
        else:
            with open(step2txt, "a") as f:
                record = str(0) + ' ' + str(int(data1[j])) + ' ' + \
                    str(int(patchx1[j])) + ' ' + str(int(patchy1[j])) + ' ' + str(int(patchx2[j])) + ' ' + str(int(patchy2[j])) + ' ' + \
                    str(NN1[j,0]) + ' ' + str(NN1[j,1]) + ' ' + str(NN1[j,2]) + ' ' + str(NN1[j,3]) + ' ' + str(0) + ' ' + str(1)  + '\n'
                f.write(record)

        ix1 = max(x1step1[j], x1step2[j])
        iy1 = max(y1step1[j], y1step2[j])
        ix2 = min(x2step1[j], x2step2[j])
        iy2 = min(y2step1[j], y2step2[j])
        IoU = (iy2-iy1+1)*(ix2-ix1+1)/((y2step2[j]-y1step2[j]+1)*(x2step2[j]-x1step2[j]+1)+(patchy2[j]-patchy1[j]+1)*(patchx2[j]-patchx1[j]+1)-(iy2-iy1+1)*(ix2-ix1+1))
        if IoU >= 0.7:
            with open(step2txt, "a") as f:
                record = str(0) + ' ' + str(int(data1[j])) + ' ' + \
                    str(int(x1step1[j])) + ' ' + str(int(y1step1[j])) + ' ' + str(int(x2step1[j])) + ' ' + str(int(y2step1[j])) + ' ' + \
                    str(NN2[j,0]) + ' ' + str(NN2[j,1]) + ' ' + str(NN2[j,2]) + ' ' + str(NN2[j,3]) + ' ' + str(1) + ' ' + str(1) + '\n'
                f.write(record)
        else:
            with open(step2txt, "a") as f:
                record = str(0) + ' ' + str(int(data1[j])) + ' ' + \
                    str(int(x1step1[j])) + ' ' + str(int(y1step1[j])) + ' ' + str(int(x2step1[j])) + ' ' + str(int(y2step1[j])) + ' ' + \
                    str(NN2[j,0]) + ' ' + str(NN2[j,1]) + ' ' + str(NN2[j,2]) + ' ' + str(NN2[j,3]) + ' ' + str(0) + ' ' + str(1) + '\n'
                f.write(record)

        with open(step2txt, "a") as f:
            record = str(0) + ' ' + str(int(data1[j])) + ' ' + \
                str(int(x1step2[j])) + ' ' + str(int(y1step2[j])) + ' ' + str(int(x2step2[j])) + ' ' + str(int(y2step2[j])) + ' ' + \
                str(NN3[j,0]) + ' ' + str(NN3[j,1]) + ' ' + str(NN3[j,2]) + ' ' + str(NN3[j,3]) + ' ' + str(1) + ' ' + str(1) + '\n'
            f.write(record)
            
    return step2txt