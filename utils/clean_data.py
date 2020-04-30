# Script to descend into a DPR folder, remove all excess
# information and resize images.
#
# Usage: python clean_data.py <dir_path> <size>
# - dir_path: path to a DPR folder containing image folders
# - size: size to rescale images to

################################################################
# CAUTION: Will permanently delete all unessesary files in dir #
################################################################

from PIL import Image
import os, sys
import argparse

def clean_data(path, size):
    # Loop through all folders in the dir
    data_folders = os.listdir(path)
    for df in data_folders:
        df_path = os.path.join(path, df)
        image_folders = os.listdir(df_path)
        for imgf in image_folders:
            imgf_path = os.path.join(df_path, imgf)
            if os.path.isdir(imgf_path):
                for item in os.listdir(imgf_path):
                    i_path = os.path.join(imgf_path, item)
                    i_name = os.path.basename(i_path)
                    f, e = os.path.splitext(i_path)
                    if imgf in i_name and (e == '.png'):
                        im = Image.open(i_path)
                        im = im.resize((int(size), int(size)))
                        im.save(f + ".jpg", 'JPEG')
                        os.remove(i_path)
                    elif (imgf not in i_name) or ('log' in i_name):
                        os.remove(i_path)


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
    return parser.parse_args()

ARGS = parse_args()
size = int(ARGS.size)
path = ARGS.dir

clean_data(path, size)