#Adam Pikielny
#Fall 2020
#analyze lighting of a face, outputting SH coordinates into a matlab file

import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

from face_detect.faceDetect import cropFace

# other modules
import os
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat
import re


# This code is adapted from https://github.com/zhhoper/DPR

def parse_args():
    parser = argparse.ArgumentParser(
        description="image relighting training.")
    parser.add_argument(
        '--model',
        default='trained.pt',
        help='model file to use stored in trained_model/'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='cpu vs. gpu'
    )
    parser.add_argument(
        '--face_detect',
        default='Neither',
        help='Options: "both" or "light". Face detection/cropping for more accurate relighting.'
    )
    parser.add_argument(
        '--videos_path',
        default='',
        help='folder with videos to put in dictionary'
    )
    parser.add_argument(
        '--fake_path',
        default='',
        help='path to fake to add to dictionary'
    )
    parser.add_argument(
        '--output_light_path',
        help='output path for sphere video'
    )
    parser.add_argument(
        '--output_sphere',
        help='output path for sphere image'
    )
    parser.add_argument(
        '--plot_path',
        help='output path for plots of SH coordinates'
    )
    parser.add_argument(
        '--frames',
        default = 30,
        help='number of frames to analyze'
    )
    parser.add_argument(
        '--mat_path',
        default='',
        help='output .mat file path'
    )
    

    return parser.parse_args()

def preprocess_image(src_img, srcOrLight):
    # src_img = cv2.imread(img_path)
    if (ARGS.face_detect == 'both') or (ARGS.face_detect == 'light' and srcOrLight == 2):
        src_img = cropFace(src_img)
    row, col, _ = src_img.shape
    src_img = cv2.resize(src_img, (256, 256))
    Lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL))
    if (ARGS.gpu):
        inputL = inputL.cuda()
    return inputL, row, col, Lab

## copied from test_network.py
def render_half_sphere(sh):
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid
    # print("outputting to ", ARGS.output_light_path)
    # cv2.imwrite(ARGS.output_light_path, shading)
    return shading


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


if (ARGS.gpu):
    sh = sh.cuda()

filePaths = os.listdir(ARGS.videos_path)
filePaths.append(ARGS.fake_path)

i = 1
dataDict = {}
for filename in filePaths:
    if (filename.endswith(".avi") or filename.endswith(".MP4")) and (re.search("camera\d.MP4", filename) is not None or re.search("camera\d.avi", filename) is not None):
        # create video reader and writer
        vc = cv2.VideoCapture(ARGS.videos_path + filename)
        _, img = vc.read()

        # i = 0
        # while img is not None:
        frames = ARGS.frames
        SHs = np.zeros((frames, 9))

        for f in range(frames):
            if img is not None:

                light_img, _, _, _ = preprocess_image(img, 2)

                sh = torch.zeros((1,9,1,1))

                _, outputSH  = my_network(light_img, sh, 0)
                # SHs.append(outputSH)
                # squashedOutput.append(torch.reshape(outputSH, (9,)).cpu().data.numpy())
                SHs[f] = torch.reshape(outputSH, (9,)).cpu().data.numpy()


                _, img = vc.read()
        if (filename == ARGS.fake_path):
            dataDict['fake'] = (SHs - np.mean(SHs)) / np.std(SHs)
        else:
            dataDict['cam' + str(i)] = (SHs - np.mean(SHs)) / np.std(SHs)
            i += 1

print(dataDict)
savemat(ARGS.mat_path, dataDict)
