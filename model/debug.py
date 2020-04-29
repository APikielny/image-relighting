import sys
sys.path.append('model')
sys.path.append('utils')

from datatime import datatime

import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse

def debug(model, epoch):
    lightFolder = 'data/example_light/'
    imgPath = 'data/obama.jpg'

    ##### getting image
    img = cv2.imread(imgPath)
    row, col, _ = img.shape
    img = cv2.resize(img, (128, 128))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL).cuda())
    ##### getting sh

    sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_02.txt'))
    sh = sh[0:9]
    sh = sh * 0.7
    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    #####

    modelId = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    saveFolder = 'result/debug/' + modelId #todo, get new debug folder if epoch is 0
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    outputImg, outputSH = model.forward(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1,2,0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg*255.0).astype(np.uint8)
    Lab[:,:,0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
    resultLab = cv2.resize(resultLab, (col, row))
    #img_name, e = os.path.splitext(ARGS.image)
    img_name = "Epoch" + epoch

    cv2.imwrite(os.path.join(saveFolder,
         '{}.jpg'.format(img_name)), resultLab)