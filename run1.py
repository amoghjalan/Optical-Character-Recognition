import pandas as pd
import numpy as np

import os
import cv2
from sys import argv

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint

from keras import backend as K

# Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(5, 5),
                 padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

model.load_weights("MNISTmodel_weights.h5")

path = argv


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = np.array(img)
            img.resize(28, 28, 1)
            images.append(img)
    return images


images = load_images_from_folder(path)

X = (images.reshape(1, 28, 28, 1).astype(np.float32))/255

with open("output.txt", 'a+') as f:
    for i in range(X.shape[0]):
        pred = model.predict(X[i])
        pred = pred.argmax()
        f.write(str(pred)+"\n")
