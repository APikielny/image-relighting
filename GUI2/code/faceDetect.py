# taken from https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# adapted by Adam Pikielny, May 2020
# plot photo with detected faces using opencv cascade classifier
import cv2
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

def cropFace(img_path):

    img = cv2.imread(img_path)

    # load the photograph
    # pixels = img
    pixels = img

    # load the pre-trained model
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)

    if len(bboxes) == 0:
        print("ERROR: No faces found.")
        return None

    # extract
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    
    BUFFER = int(width * 0.25)

    images = []

    # show the image
    for i in range(len(bboxes)):
        x, y, width, height = bboxes[i]
        x2, y2 = x + width, y + height
        images.append(pixels[max(y - BUFFER, 0):min(y2 + BUFFER, pixels.shape[0]), max(x - BUFFER, 0):min(x2 + BUFFER, pixels.shape[1])])
        # imshow('hehe', images[i])
        # waitKey(0)
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    return images

