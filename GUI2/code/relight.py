'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse

from model import *

class Relight():
    def __init__(self, source, light, dest):

        self.relighting(source, light, dest)


    def preprocess_image(self, img):
        row, col, _ = img.shape
        src_img = cv2.resize(img, (256, 256))
        Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB) 

        inputL = Lab[:,:,0] 
        inputL = inputL.astype(np.float32)/255.0
        inputL = inputL.transpose((0,1))
        inputL = inputL[None,None,...]
        inputL = Variable(torch.from_numpy(inputL))

        return inputL, row, col, Lab

    def relighting(self, source, light, dest):

        # load model
        my_network = HourglassNet()

        my_network.load_state_dict(torch.load('trained_models/model256.pt', map_location=torch.device('cpu')))

        my_network.train(False)

        # saveFolder = os.path.join(saveFolder, source_path.split(".")[0])

        light_img, _, _, _ = self.preprocess_image(light)

        sh = torch.zeros((1,9,1,1))

        _, outputSH  = my_network(light_img, sh, 0)

        src_img, row, col, Lab = self.preprocess_image(source)

        outputImg, _ = my_network(src_img, outputSH, 0)

        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        
        cv2.imwrite(os.path.join(dest,
            'relit.jpg'), resultLab)






# ARGS = parse_args()

# modelFolder = 'trained_models/'

# # load model
# from model import *
# my_network = HourglassNet()

# if (ARGS.gpu):
#     my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model)))
#     my_network.cuda()
# else:
#     my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model), map_location=torch.device('cpu')))

# my_network.train(False)

# saveFolder = 'result'
# saveFolder = os.path.join(saveFolder, ARGS.model.split(".")[0])
# if not os.path.exists(saveFolder):
#     os.makedirs(saveFolder)

# light_img, _, _, _ = preprocess_image('data/{}'.format(ARGS.light_image), 2)

# sh = torch.zeros((1,9,1,1))
# if (ARGS.gpu):
#     sh = sh.cuda()

# _, outputSH  = my_network(light_img, sh, 0)

# src_img, row, col, Lab = preprocess_image('data/{}'.format(ARGS.source_image), 1)

# outputImg, _ = my_network(src_img, outputSH, 0)

# outputImg = outputImg[0].cpu().data.numpy()
# outputImg = outputImg.transpose((1,2,0))
# outputImg = np.squeeze(outputImg)
# outputImg = (outputImg*255.0).astype(np.uint8)
# Lab[:,:,0] = outputImg
# resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
# resultLab = cv2.resize(resultLab, (col, row))
# img_name, e = os.path.splitext(ARGS.source_image)
# if (ARGS.face_detect == 'both'):
#     img_name += "_faceDetectBoth"
# if (ARGS.face_detect == 'light'):
#     img_name += "_faceDetectLight"
# cv2.imwrite(os.path.join(saveFolder,
#         '{}_relit.jpg'.format(img_name)), resultLab)
# #----------------------------------------------