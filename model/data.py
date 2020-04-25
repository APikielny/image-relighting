from torch.autograd import Variable
import cv2
import torch
import numpy as np

class ImagePair:
    def __init__(self, I_s, I_t, L_s, L_t):
        # Source image
        self.I_s = I_s

        # Target image
        self.I_t = I_t

        # Source lighting
        self.L_s = L_s

        # Target lighting
        self.L_t = L_t

# Loads all the data from the dataset at "path" into a Python list
# of ImagePairs
# From each image folder, pull 2 image pairs with corresponding lighting
# shapes/formats found in testNetwork_demo_512.py
def load_data(path):
    return None