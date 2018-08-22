# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:54:39 2018

@author: wufei
"""


import _init_paths
import os.path as osp
import shutil
import numpy as np
import random
import cv2
from PIL import Image, ImageEnhance
from detect.config import cfg


def augment_data(filename):

    '''
    Image Augmentation
    
    Parameter
        filename : the path of the training images (before augmentation)
    '''    

    # the path of the original images(before augmentation)
    srcImagePath = filename

    # the path of the training images (after augmentation)
    dstImagePath = cfg.TRAIN_IMAGE_DIR

#    if not osp.exists(dstImagePath):
#        os.mkdir(dstImagePath)

    if not osp.exists(dstImagePath):
        shutil.copytree(srcImagePath, dstImagePath)
    else:
        print('The path of the augmented images exists. If you want to augment the images again, please delete the folder containing augmented images, and run this function again.')
        print('The corresponing path is ' + dstImagePath)
        exit()

    '''
    here i1, i2, i3 denote the total number of images before augmentation, 
    the total number of images after fliping, the total number of images after scaling, respectively 
    '''
    i1 = 200
    i2 = 600
    i3 = 1200

    for index in range(1, i1 + 1):
        imagePath  = osp.join(dstImagePath, str(index) + cfg.IMAGE_TYPE)
        # here the recordPath refers to the path of txt file which saves the position of ground-truth bounding box and label
        recordPath = osp.join(dstImagePath, str(index) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):
        #data1,  data2,  data3,  data4,  data5,  data6,  data7,  data101,  data102,  data103  data104
        #起点x   起点y   终点x   终点y   类别    中点x   中点y   框左上x   框左上y   框右下x  框右下y
            '''
            data1, data2: the position of ship head
            data3, data4: the position of stern
            data5: label
            data101, data102: the position of up-left point of bounding box
            data103  data104: the position of down-right point of bounding box
            data6, data7: the position of midpoint of bounding box
                and data6 = (data101+ data103)/2, data7 = (data102+ data104)/2
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
                data1.append(int(data1_tmp))
                data2.append(int(data2_tmp))
                data3.append(int(data3_tmp))
                data4.append(int(data4_tmp))
                data5.append(int(data5_tmp))
                data6.append(int((data101_tmp+data103_tmp)/2.))
                data7.append(int((data102_tmp+data104_tmp)/2.))
                data101.append(int(min(data101_tmp, data103_tmp)))
                data102.append(int(min(data102_tmp, data104_tmp)))
                data103.append(int(max(data101_tmp, data103_tmp)))
                data104.append(int(max(data102_tmp, data104_tmp)))

            shipNumber = len(data1)
            im = Image.open(imagePath)

            #augmentation sterp 1 : left-right flipping and
            im1 = im.transpose(Image.FLIP_LEFT_RIGHT)
            print (osp.join(dstImagePath, str(index + i1) + cfg.IMAGE_TYPE))
            im1.save(osp.join(dstImagePath, str(index + i1) + cfg.IMAGE_TYPE))
            for j in range(0, shipNumber):
                with open(osp.join(dstImagePath, str(i1 + index) + '.txt'), 'a') as f:
                    record = str(cfg.IMAGE_WIDTH + 1 - data1[j]) + ' ' + str(data2[j]) + ' ' + str(cfg.IMAGE_WIDTH + 1 - data3[j]) + ' ' + str(data4[j]) + ' ' + \
                        str(data5[j]) + ' ' + str(cfg.IMAGE_WIDTH + 1 - data6[j]) + ' ' + str(data7[j]) + ' ' + \
                        str(cfg.IMAGE_WIDTH + 1 - data103[j]) + ' ' + str(data102[j]) + ' ' + \
                        str(cfg.IMAGE_WIDTH + 1 - data101[j]) + ' ' + str(data104[j]) + '\n'
                    f.write(record)

            #augmentation sterp 1 : top-down flipping
            im2 = im.transpose(Image.FLIP_TOP_BOTTOM)
            print (osp.join(dstImagePath, str(index + i1 * 2) + cfg.IMAGE_TYPE))
            im2.save(osp.join(dstImagePath, str(index + i1 * 2) + cfg.IMAGE_TYPE))
            for j in range(0, shipNumber):
                with open(osp.join(dstImagePath, str(i1 * 2 + index) + '.txt'), 'a') as f:
                    record = str(data1[j]) + ' ' + str(cfg.IMAGE_HEIGHT + 1 - data2[j]) + ' ' + str(data3[j]) + ' ' + str(cfg.IMAGE_HEIGHT + 1 - data4[j]) + ' ' + \
                        str(data5[j]) + ' ' + str(data6[j]) + ' ' + str(cfg.IMAGE_HEIGHT + 1 - data7[j]) + ' ' + \
                        str(data101[j]) + ' ' + str(cfg.IMAGE_HEIGHT + 1 - data104[j]) + ' ' + \
                        str(data103[j]) + ' ' + str(cfg.IMAGE_HEIGHT + 1 - data102[j]) + '\n'
                    f.write(record)


    #augmentation sterp 2 : scaling
    for index in range(1, i2 + 1):
        imagePath  = osp.join(dstImagePath, str(index) + cfg.IMAGE_TYPE)
        recordPath = osp.join(dstImagePath, str(index) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):
            data1 = []
            data2 = []
            data3 = []
            data4 = []
            data5 = []
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
                data1.append(data1_tmp)
                data2.append(data2_tmp)
                data3.append(data3_tmp)
                data4.append(data4_tmp)
                data5.append(data5_tmp)
                data101.append(min(data101_tmp, data103_tmp))
                data102.append(min(data102_tmp, data104_tmp))
                data103.append(max(data101_tmp, data103_tmp))
                data104.append(max(data102_tmp, data104_tmp))
     
            data1 = np.array(data1, dtype = float)
            data2 = np.array(data2, dtype = float)
            data3 = np.array(data3, dtype = float)
            data4 = np.array(data4, dtype = float)
            data5 = np.array(data5, dtype = int)
            data101 = np.array(data101, dtype = float)
            data102 = np.array(data102, dtype = float)
            data103 = np.array(data103, dtype = float)
            data104 = np.array(data104, dtype = float)

            shipNumber = len(data1)
            im = cv2.imread(imagePath)

            #scaling factor 0.8 ~ 1.2
            scale = random.uniform(0.8, 1.2)

            height, width, channels = im.shape
            im3 = cv2.resize(im, (int(scale*width), int(scale*height)), interpolation=cv2.INTER_CUBIC)

            radius = int(max(height, width)/2.)

            im4 = cv2.copyMakeBorder(im3, radius, radius, radius, radius, cv2.BORDER_REPLICATE)
            height, width, channels = im4.shape
            centerx = int(width/2.)
            centery = int(height/2.)

            im5 = im4[centery-cfg.IMAGE_HEIGHT/2:centery+cfg.IMAGE_HEIGHT/2, centerx-cfg.IMAGE_WIDTH/2:centerx+cfg.IMAGE_WIDTH/2, :]

            offsetx = centerx - cfg.IMAGE_WIDTH/2
            offsety = centery - cfg.IMAGE_HEIGHT/2

            data1[:] = data1[:] * scale + radius - offsetx
            data2[:] = data2[:] * scale + radius - offsety 
            data3[:] = data3[:] * scale + radius - offsetx
            data4[:] = data4[:] * scale + radius - offsety
            data101[:] = data101[:] * scale + radius - offsetx
            data102[:] = data102[:] * scale + radius - offsety
            data103[:] = data103[:] * scale + radius - offsetx
            data104[:] = data104[:] * scale + radius - offsety

            print(osp.join(dstImagePath, str(index + i2) + cfg.IMAGE_TYPE))
            cv2.imwrite(osp.join(dstImagePath, str(index + i2) + cfg.IMAGE_TYPE), im5)

            for j in range(0, shipNumber):
                with open(osp.join(dstImagePath, str(i2 + index) + '.txt'), "a") as f:
                    record = str(int(data1[j])) + ' ' + str(int(data2[j])) + ' ' + str(int(data3[j])) + ' ' + str(int(data4[j])) + ' ' + \
                        str(data5[j]) + ' ' + str(int((data101[j] + data103[j])/2.)) + ' ' + str(int((data102[j] + data104[j])/2.)) + ' ' + \
                        str(int(data101[j])) + ' ' + str(int(data102[j])) + ' ' + \
                        str(int(data103[j])) + ' ' + str(int(data104[j])) + '\n'
                    f.write(record)

    #augmentation step 3
    for index in range(1, i3 + 1):
        imagePath  = osp.join(dstImagePath, str(index) + cfg.IMAGE_TYPE)
        recordPath = osp.join(dstImagePath, str(index) + '.txt')
        if(osp.exists(recordPath) and osp.exists(imagePath)):

            image = Image.open(imagePath)

            # saturation
            random_factor = np.random.randint(6, 15) / 10.
            color_image = ImageEnhance.Color(image).enhance(random_factor)
            
            # brightness
            random_factor = np.random.randint(6, 15) / 10.
            brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
            
            # contrast
            random_factor = np.random.randint(6, 15) / 10.
            contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
            
            # sharpness
            random_factor = np.random.randint(6, 15) / 10.
            imagenew =  ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

            print (osp.join(dstImagePath, str(i3 + index) + cfg.IMAGE_TYPE))
            imagenew.save(osp.join(dstImagePath, str(i3 + index) + cfg.IMAGE_TYPE))


if __name__ == '__main__':

    image_dir = osp.join(cfg.ROOT_DIR + '/dataset/train')

    augment_data(image_dir)
    
    print ('Done.')