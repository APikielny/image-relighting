import sys
sys.path.append('model')
sys.path.append('utils')

from datetime import datetime

import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse

def test(model, modelId = None):
    lightFolder = '../data/example_light/'
    imgPath = '../data/obama.jpg'

    ##### getting image
    img = cv2.imread(imgPath)
    row, col = img.shape
    img = cv2.resize(img, (128, 128))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL).cuda())

    saveFolder = '../result/test/' + modelId
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    
    for i in range(7):
        ##### getting sh

        sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
        sh = sh[0:9]
        sh = sh * 0.7
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        #####

        outputImg, outputSH = model.forward(inputL, sh, 0)
        outputImg = outputImg[0].cpu().data.numpy()
        outputImg = outputImg.transpose((1,2,0))
        outputImg = np.squeeze(outputImg)
        outputImg = (outputImg*255.0).astype(np.uint8)
        Lab[:,:,0] = outputImg
        resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
        resultLab = cv2.resize(resultLab, (col, row))
        #img_name, e = os.path.splitext(ARGS.image)
        img_name =  "Light" + '{:02}'.format(i)

        cv2.imwrite(os.path.join(saveFolder,
            '{}.jpg'.format(img_name)), resultLab)

    return modelId

def parse_args():
    parser = argparse.ArgumentParser(
        description="image relighting training.")
    parser.add_argument(
        '--image',
        default='obama.jpg',
        help='name of image stored in data/',
    )
    parser.add_argument(
        '--model',
        default='model_9.pt',
        help='model file to use stored in trained_model/'
    )

    return parser.parse_args()


ARGS = parse_args()
modelFolder = '../trained_models/'

from model import *
my_network = HourglassNet()
my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model)))
my_network.cuda()
my_network.train(False)

test(my_network)
