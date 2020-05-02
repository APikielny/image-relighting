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


def parse_args():
    parser = argparse.ArgumentParser(
        description="image relighting training.")
    parser.add_argument(
        '--source_image',
        default='obama.jpg',
        help='name of image stored in data/',
    )
    parser.add_argument(
        '--light_image',
        default='obama.jpg',
        help='name of image stored in data/',
    )
    parser.add_argument(
        '--model',
        default='default.pt',
        help='model file to use stored in trained_model/'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='cpu vs. gpu'
    )

    return parser.parse_args()

def preprocess_image(img_path):
    src_img = cv2.imread(img_path)
    row, col, _ = src_img.shape
    src_img = cv2.resize(src_img, (128, 128))
    Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL))
    if (ARGS.gpu):
        inputL = inputL.cuda()
    return inputL, row, col, Lab


ARGS = parse_args()

modelFolder = 'trained_models/'

# load model
from model import *
my_network = HourglassNet()

if (ARGS.gpu):
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model)))
    my_network.cuda()
else:
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, ARGS.model), map_location=torch.device('cpu')))

my_network.train(False)

lightFolder = 'data/example_light/'

saveFolder = 'result'
saveFolder = os.path.join(saveFolder, ARGS.model.split(".")[0])
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

light_img, _, _, _ = preprocess_image('data/{}'.format(ARGS.light_image))

sh = torch.zeros((1,9,1,1))
if (ARGS.gpu):
    sh = sh.cuda()

_, outputSH  = my_network(light_img, sh, 0)

src_img, row, col, Lab = preprocess_image('data/{}'.format(ARGS.source_image))

outputImg, _ = my_network(src_img, outputSH, 0)

outputImg = outputImg[0].cpu().data.numpy()
outputImg = outputImg.transpose((1,2,0))
outputImg = np.squeeze(outputImg)
outputImg = (outputImg*255.0).astype(np.uint8)
Lab[:,:,0] = outputImg
resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
resultLab = cv2.resize(resultLab, (col, row))
img_name, e = os.path.splitext(ARGS.source_image)
cv2.imwrite(os.path.join(saveFolder,
        '{}_relit.jpg'.format(img_name)), resultLab)
#----------------------------------------------