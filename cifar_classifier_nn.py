# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:38:15 2021

@author: diggee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

#%% restricting tensorflow GPU usage

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)   # prevents tf from allocating entire GPU RAM upfront
  except RuntimeError as e:
    print(e)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   #to force tf to only use CPU and not GPU 

#%% Function to extract data from CIFAR10 files

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%% Reshaping image to 3D array

def rearrange_image(image_array, n_size):
    new_image_array = np.zeros((n_size, n_size, 3))
    count = 0
    for i in range(3):
        for j in range(n_size):
            for k in range(n_size):
                new_image_array[j,k,i] = image_array[n_size*count + k]
            count += 1
    return new_image_array

#%% Processing data to feed to neural network

def preprocess_data(train_data, test_data):    
    X = np.empty((50000, 32, 32, 3), float)
    y = np.empty((50000, 1), int)
    counter = 0
    for data in train_data:
        for i in range(len(data[b'data'])):
            X[counter*len(data[b'data']) + i] = rearrange_image(data[b'data'][i], 32)
            y[counter*len(data[b'labels']) + i] = data[b'labels'][i]
        counter += 1
    y = to_categorical(y)
    
    X_test = np.empty((10000, 32, 32, 3), float)
    counter = 0
    for i in range(len(test_data[b'data'])):
        X_test[counter*len(test_data[b'data']) + i] = rearrange_image(test_data[b'data'][i], 32)
    counter += 1
    
    return X, y, X_test

#%% Neural Network

def neural_network(X, y, validation_split, n_filters, kernel_size):
    model = Sequential() 
    model.add(Conv2D(filters = n_filters, kernel_size = kernel_size, activation = 'relu', padding = 'same', input_shape = (32,32,3), data_format = 'channels_last'))
    # model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.1))
        
    model.add(Conv2D(filters = n_filters*2, kernel_size = kernel_size, activation = 'relu', padding = 'same'))
    # model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters = n_filters*4, kernel_size = kernel_size, activation = 'relu', padding = 'same'))
    # model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters = n_filters*8, kernel_size = kernel_size, activation = 'relu', padding = 'same'))
    # model.add(MaxPooling2D(2, strides = 1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(n_filters*8, 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(n_filters*4, 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(10, 'softmax'))
    model.summary()    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    stop_early = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min', verbose = 1, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = 0.00001, verbose = 1)
    
    train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      rescale = 1/255,
      fill_mode='nearest',
      validation_split = validation_split
      )

    train_generator = train_datagen.flow(X, y, batch_size = 128, subset = 'training')
    valid_generator = train_datagen.flow(X, y, batch_size = 128, subset = 'validation')
    history = model.fit(train_generator, epochs = 300, verbose = 1, validation_data = valid_generator, callbacks = [stop_early, checkpoint, reduce_lr])
    return model, history

#%% Transfer learning neural network

def ResNet50NN(X, y, validation_split):       
    X = preprocess_input(X)
        
    resnet50_base_model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
    resnet50_base_model.trainable = False
    inputs = tf.keras.Input(shape = (32, 32, 3))
    resized = tf.keras.layers.UpSampling2D(size = (7, 7))(inputs)
    
    x = resnet50_base_model(resized, training = False)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    classification_output = Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs = inputs, outputs = classification_output)
    
    model.summary()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    stop_early = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min', verbose = 1, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, min_lr = 0, verbose = 1)
    
    history = model.fit(X, y, epochs = 300, batch_size = 64, verbose = 1, validation_split = validation_split, callbacks = [stop_early, checkpoint, reduce_lr])
    return model, history

#%% Plots

def make_plots(history):
    
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    
#%% test predictions

def test_prediction(model, n_pixels):
    images = [] 
    for file in sorted(os.listdir('test'), key = len):
        image = img_to_array(load_img('test/' + file, target_size = (n_pixels, n_pixels)), dtype = 'uint8')
        image = image/255
        image = np.reshape(image, (-1, n_pixels, n_pixels, 3))
        # image = preprocess_input(image)
        images.append(image)
    images = np.vstack(images)    
    predictions = model.predict_classes(images, batch_size = 128, verbose = 1)
    # predictions = model.predict(images, batch_size = 128, verbose = 1)
    # predictions = predictions.argmax(axis = 1)
    return predictions
    
#%% main

train_data = []
files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
for file in files:
    train_data.append(unpickle(file))
test_data = unpickle('test_batch')
X, y, X_test = preprocess_data(train_data, test_data)

# model, history = ResNet50NN(X, y, 0.05)

models = []
histories = []
n_filters=  [128]
kernel_size = [3]
for i in n_filters:
    for j in kernel_size:
        print('no of filters = ', i,', kernel size = ', j)
        model, history = neural_network(X, y, 0.05, i, j)
        models.append(model)
        histories.append(history)
        make_plots(history)
        predictions = model.predict_classes(X_test/255)
        print(classification_report(test_data[b'labels'], predictions))
        print(confusion_matrix(test_data[b'labels'], predictions))

# uncomment the following to predict with ResNet based model
# predictions = model.predict(preprocess_input(X_test))
# predictions = predictions.argmax(axis = 1)
# print(classification_report(test_data[b'labels'], predictions))
# print(confusion_matrix(test_data[b'labels'], predictions))

df = pd.read_csv('sampleSubmission.csv')
df.label = test_prediction(models[0], 32)      
cifar10_classes = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
df.label = df.label.apply(lambda x: cifar10_classes.get(x))
df.to_csv("sampleSubmission.csv", index = False)