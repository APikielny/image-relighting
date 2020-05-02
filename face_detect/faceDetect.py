# taken from https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# adapted by Adam Pikielny, May 2020
# plot photo with detected faces using opencv cascade classifier
from cv2 import imread, resize
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

def cropFace(img):

    # load the photograph
    # pixels = img
    pixels = img

    # load the pre-trained model
    classifier = CascadeClassifier('face_detect/haarcascade_frontalface_default.xml')

    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)

    # extract
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    
    BUFFER = int(width * 0.25)

    # show the image
    image = pixels[max(y - BUFFER, 0):min(y2 + BUFFER, pixels.shape[0]), max(x - BUFFER, 0):min(x2 + BUFFER, pixels.shape[1])]
    # imshow('hehe', image)
    # waitKey(0)
    return image

#cropFace(1)
