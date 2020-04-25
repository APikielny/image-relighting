import os

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
    img_pairs = []

    for i in range(2):
        folder_path = os.path.join(path, '../data/dpr_{:2d}'.format(i*5000))
        img_folders = next(os.walk(folder_path))[1]
        for img_folder in img_folders:
            pair = np.random.choice(5, 2)

            image_s_path = os.path.join(folder_path, img_folder, image_1_name, img_folder + "_0" + pair[0] + ".jpg")
            image_t_path = os.path.join(folder_path, img_folder, image_1_name, img_folder + "_0" + pair[1] + ".jpg")
            lighting_s_path = os.path.join(folder_path, img_folder, image_1_name, img_folder + "_light_0" + pair[0] + ".txt")
            lighting_t_path = os.path.join(folder_path, img_folder, image_1_name, img_folder + "_light_0" + pair[1] + ".txt")
            print(image_s_path)
            print(image_t_path)
            print(lighting_s_path)
            print(lighting_t_path)

            I_s = get_image(image_s_path)
            I_t = get_image(image_t_path)
            L_s = get_lighting(lighting_s_path)
            L_t = get_lighting(lighing_t_path)
            img_pair = new ImagePair(I_s, I_t, L_s, L_t)
            img_pairs.append(img_pair)
            print("-----------------------------------------------------------")
            print("-----------------------------------------------------------")
        


    return np.asarray(img_pairs)

# will make pulling multiple image pairs from the same folder nice
# def make_image_pair(folder_path):
#     #randomly select two images and their lighting
#     #pass image paths to get_image, and lighting paths to get_lighting
#     img_pair = new ImagePair(I_s, I_t, L_s, L_t)
#     return img_pair

def get_image(path_to_image):
    img = cv2.imread(path_to_image)
    # row, col, _ = img.shape
    # img = cv2.resize(img, (512, 512))
    img = cv2.resize(img, (128, 128)) #we changed this because of reduced image size
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #converts image to one color space LAB

    inputL = Lab[:,:,0] #taking only the L channel
    inputL = inputL.astype(np.float32)/255.0 #normalise
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...] #not sure what's happening here
    inputL = Variable(torch.from_numpy(inputL).cuda())
    
    return inputL
    

def get_lighting(path_to_light):
    sh = np.loadtxt(os.path.join(lightFolder, path_to_light))
    sh = sh[0:9]
    sh = sh * 0.7

    sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    return sh


load_data("")