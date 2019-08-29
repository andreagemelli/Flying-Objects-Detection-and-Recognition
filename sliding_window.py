from keras.applications.resnet50 import ResNet50
from keras.models import Model
import cv2
from keras.optimizers import Adam, Nadam
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
import math

from model import model


def sliding_window(im, tmp, index):
    windows = []
    x_coords = []
    y_coords = []
    print("immagine : " + im)
    im_frame = cv2.imread(im)
    im_frame = cv2.cvtColor(im_frame, cv2.COLOR_BGR2GRAY)
    count = 0
    if True:
        for x in range(0, im_frame.shape[1] - w_width, stepSize):
            for y in range(0, im_frame.shape[0] - w_height, stepSize):
                window = im_frame[x:x + w_width, y:y + w_height]

                if window.shape[0] != 0 and window.shape[1] != 0:
                    
                    window = cv2.resize(window, (40, 40))
                    cv2.imwrite('windows/window'+str(count)+'.jpg',window)
                    window = np.reshape(window, [1, 40, 40, 1])
                    window = window / 255.
                    windows.append(window)
                    x_coords.append(x)
                    y_coords.append(y)
                    
                    count += 1

        windows = np.vstack(windows)
        predictions = model.predict(windows, steps = 1)
        print(max(predictions))
        for i in range(len(predictions)):
            if predictions[i] >= 0.16:
              
                x = x_coords[i]
                y = y_coords[i]
                print(predictions[i])
                print(x)
                print(y)
                cv2.rectangle(tmp, (y, x), (y + w_height, x + w_width), (0, 255, 0), 2)  # draw rectangle on image
            
    cv2.imshow('Frame', tmp)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    model = model(img_height=40, img_width=40, num_classes=1, lr=0.0001)
    model.load_weights('weights/vgg19_binary.h5')

    stepSize = 40
    (w_width, w_height) = (40, 40)  # window size
    index = 0

    for img in sorted(os.listdir("video_test")):
        im = os.path.join("video_test", img)
        tmp = cv2.imread(im)
        sliding_window(im, tmp, index)
        index += 1


