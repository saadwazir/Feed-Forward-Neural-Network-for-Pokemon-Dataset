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



'''
lst = joblib.load('expt-images')

#print(lst[2])

dataset = pd.read_csv("pokemon-dataset.csv")    # reading the csv file
dataset.columns = ['File-ID','label']

print(dataset)

lst

pd = dataset
for i in range(  len(lst) ):
    index = pd[ pd['File-ID'] == lst[i] ].index
    pd.drop(index, inplace=True)

print(pd)

export_csv = pd.to_csv(r'pokemon-dataset-new.csv', index = None, header=False)

'''


X1 = joblib.load('pokemon-images')
print(X1.shape)
dataset = pd.read_csv("pokemon-dataset-new.csv")
print(dataset.shape)






