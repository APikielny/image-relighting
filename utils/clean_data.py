# Script to descend into a DPR folder, remove all excess
# information and resize images.
#
# Usage: python clean_data.py --dir <dir_path> --size <size> --save <save_path>
# - dir_path: path to a DPR folder containing image folders
# - size: size to rescale images to
# - save_path: path to a folder to save all image folders

from PIL import Image
import os, shutil
import argparse

def clean_data(path, size, save):
    # Loop through all folders in the dir
    image_folders = os.listdir(path)
    for imgf in image_folders:
        imgf_path = os.path.join(path, imgf)
        if os.path.isdir(imgf_path):
            save_path = os.path.join(save, imgf)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for item in os.listdir(imgf_path):
                i_path = os.path.join(imgf_path, item)
                i_name = os.path.basename(i_path)

                if (imgf in i_name) and ('.png' in i_name):
                    im = Image.open(i_path)
                    im = im.resize((size, size))
                    im.save(os.path.join(save_path, (i_name[:len(i_name)-4] + ".jpg")), 'JPEG')
                elif '_light_' in i_name:
                    shutil.copyfile(i_path, os.path.join(save_path, i_name))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to clean data for image-relighting")
    parser.add_argument(
        '--size',
        required=True,
        type=int,
        help='the size to rescale images to')
    parser.add_argument(
        '--dir',
        required=True,
        help='the directory containing all the data'
    )
    parser.add_argument(
        '--save',
        required=True,
        help='path to save data'
    )
    return parser.parse_args()

ARGS = parse_args()
size = ARGS.size
path = ARGS.dir
save  = ARGS.save

clean_data(path, size, save)