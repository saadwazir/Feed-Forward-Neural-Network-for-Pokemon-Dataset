

import time
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler,ReduceLROnPlateau
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tensorflow as tf
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image
from matplotlib import pyplot
import feedforward_keras as fk
import keras
from keras.utils import np_utils
from keras.datasets import cifar10
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from numpy import load
from tqdm import tqdm
import os
import pandas as pd
import csv
import time
import joblib
from sklearn.model_selection import train_test_split

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))





x = joblib.load('pokemon-images')
print(x.shape)
x = np.reshape(x, (2327, 3072))
print(x.shape)

dataset = pd.read_csv("pokemon-dataset-new.csv")    # reading the csv file
dataset.columns = ['File-ID','label']
y = dataset.drop(['File-ID'], axis = 1)
print(y.shape)
y_ = keras.utils.to_categorical(y['label'], 5)
print(y_)


X_train, X_test, y_train, y_test = train_test_split( x, y_, test_size=0.3, random_state=42)

print("--------------------------------")
print( X_train.shape )
print( X_test.shape )
print( y_train.shape )
print( y_test.shape )



print ('Data loaded.')
data =  [X_train, X_test, y_train, y_test]


def model():
    start_time = time.time()
    print('Compiling Model ... ')
    model = Sequential()

    model.add(Dense(5, input_dim=3072))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))

    model.add(Dense(500*100))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.3))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    sgd = SGD(learning_rate=0.01)
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print('Model compield in {0} seconds'.format(time.time() - start_time))
    return model


model, losses = fk.run_network(data, model(), 100, 16)



'''

y_train = keras.utils.to_categorical(train['label'], 2)
y_test = keras.utils.to_categorical(test['label'], 2)

X_train /= 255
X_test /= 255

X_train = np.reshape(X_train, (24044, 361))
X_test = np.reshape(X_test, (6976, 361))


print ('Data loaded.')
data =  [X_train, X_test, y_train, y_test]




fk.plot_losses(losses)

'''