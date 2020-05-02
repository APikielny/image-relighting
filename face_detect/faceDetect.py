# taken from https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
# adapted by Adam Pikielny, May 2020
# plot photo with detected faces using opencv cascade classifier
from cv2 import imread, resize
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

BUFFER = 50

def cropFace(img_path):

    # load the photograph
    img = imread('../data/boys.jpg')
    pixels = resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    # load the pre-trained model
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)
    # print bounding box for each detected face

    # extract
    x, y, width, height = bboxes[0]
    x2, y2 = x + width, y + height
    # draw a rectangle over the pixels

    # show the image
    return pixels[y - BUFFER:y2 + BUFFER, x - BUFFER:x2 + BUFFER]
