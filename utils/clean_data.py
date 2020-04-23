# Descend into DPR Folder
# In each image folder:
#   - Delete all files that are not images or lighting
#   - Resize all images to input size

from PIL import Image
import os, sys

def clean_data(path, size):
    # Loop through all folders in the dir
    folders = os.listdir(path)
    for folder in folders:
        f_path = path+folder
        if os.path.isdir(f_path):

            # Loop through all files in the folder
            for item in os.listdir(f_path):
                i_path = f_path+"/"+item
                i_name = os.path.basename(i_path)
                f, e = os.path.splitext(i_path)

                # Resize the 5 lighting images and remove
                # the extra files
                if folder in i_name and (e == ".png"):
                    im = Image.open(i_path)
                    im = im.resize((int(size), int(size)))
                    im.save(f + ".jpg", 'JPEG')
                    os.remove(i_path)
                elif folder not in i_name:
                    os.remove(i_path)

if (len(sys.argv) != 3):
    print("usage: <dir_path> <resize>")
else:
    path = sys.argv[1]
    size  = sys.argv[2]
    if not os.path.isdir(path):
        print("error: invalid directory")
    elif not size.isdigit():
        print("error: invalid size")
    else:
        dir_name = os.path.basename(path)
        print("cleaning folder: " + dir_name)
        print("resizing images to size: " + size)

        size = int(size)
        if path[-1] != "/":
            path = path + "/"
        print(path)
        
        clean_data(path, size)
