import numpy as np
import cv2
from torch.autograd import Variable
import torch
import os
from torch.utils.data import Dataset

class CelebData(Dataset):
    def __init__(self, root_dir, max_data):
        paths = []
        for i in range(6):
            folder_path = os.path.join(root_dir, 'dpr_{:d}'.format(i * 5000))
            img_folders = os.listdir(folder_path)
            for img_folder in img_folders:
                path = os.path.join(folder_path, img_folder)
                if os.path.isdir(path):
                    paths.append(path)
                    if len(paths) == max_data:
                        break
            if len(paths) == max_data:
                break

        add = max_data - len(paths)
        if add <= 0:
            self.paths = paths
        else:
            self.paths = loop_data_helper(paths, add)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        folder_path = self.paths[idx]
        pair = np.random.choice(5, 2)
        img_folder_name = folder_path[-10:]

        image_s_path = os.path.join(folder_path, img_folder_name + "_0" + str(pair[0]) + ".jpg")
        image_t_path = os.path.join(folder_path, img_folder_name + "_0" + str(pair[1]) + ".jpg")
        lighting_s_path = os.path.join(folder_path, img_folder_name + "_light_0" + str(pair[0]) + ".txt")
        lighting_t_path = os.path.join(folder_path, img_folder_name + "_light_0" + str(pair[1]) + ".txt")

        I_s = get_image(image_s_path)
        I_t = get_image(image_t_path)
        L_s = get_lighting(lighting_s_path)
        L_t = get_lighting(lighting_t_path)

        return I_s, I_t, L_s, L_t


def get_image(path_to_image):
    img = cv2.imread(path_to_image)
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # converts image to one color space LAB

    inputL = Lab[:, :, 0]  # taking only the L channel
    inputL = inputL.astype(np.float32) / 255.0  # normalise
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]  # not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL))

    return inputL


def get_lighting(path_to_light):
    sh = np.loadtxt(path_to_light)
    sh = sh[0:9]
    sh = sh * 0.7

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh))
    return sh

def loop_data_helper(paths, add_dup):
    data_size = len(paths)
    orig_paths = paths.copy()

    num_add = add_dup // data_size
    for i in range(num_add):
        paths.extend(orig_paths)
    rem = add_dup % data_size
    paths.extend(orig_paths[0:rem])

    return paths
