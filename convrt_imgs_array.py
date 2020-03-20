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
import joblib


dataset = pd.read_csv("pokemon-dataset-new.csv")    # reading the csv file
dataset.columns = ['File-ID','label']

print(dataset)

def conv_imgs(pd):
    train_image = []
    expt = []
    for i in tqdm(range(pd.shape[0])):
        try:
            img_load = Image.open("/home/saad/pokemon_dataset/" + pd['File-ID'][i])
            img_load = img_load.convert('RGBA')
            #img_load = img_load.convert('RGB')
            img_load.thumbnail((32, 32), Image.ANTIALIAS)
            padImg = Image.new('RGB', (32, 32), (255, 255, 255))
            padImg.paste(img_load)
            img = img_to_array(padImg)
            img = img.astype('float32')
            img /= 255.0
            train_image.append(img)
        except Exception:
            expt.append(pd['File-ID'][i])
            pass

    X = np.array(train_image)
    return X, expt


X, e = conv_imgs(dataset)


joblib.dump(X, 'pokemon-images')
X1 = joblib.load('pokemon-images')

print(X1.shape)

joblib.dump(e, 'expt_imgs')
e1 = joblib.load('expt_imgs')
print(e1)
print(len(e1))