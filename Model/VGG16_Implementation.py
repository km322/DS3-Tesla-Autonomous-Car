import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from keras.applications import vgg16
from keras.models import Model,load_model
import keras
from keras.layers import Input, Conv2D, Conv2DTranspose,AveragePooling2D, MaxPooling2D,UpSampling2D,LeakyReLU, concatenate, Dropout,BatchNormalization,Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K





def create_segmentation_model(input_shape, num_classes):
    base_model = vgg16.VGG16(include_top=False, input_shape=input_shape, weights=VGG16_weight)
    
    #uncomment this if want to disable back propagation
    # for layer in base_model.layers:
    #     layer.trainable = False
    
    #encoder
    encoder_output = base_model.output
    
    #decoder
    '''first upsample'''
    x = UpSampling2D(size=(2, 2))(encoder_output)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    skip_connection_layer = base_model.get_layer('block5_conv3').output
    x = concatenate([x, skip_connection_layer])
    
    '''Second upsample'''
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    skip_connection_layer = base_model.get_layer('block4_conv3').output
    x = concatenate([x, skip_connection_layer])
    
    '''third upsample'''
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    skip_connection_layer = base_model.get_layer('block3_conv3').output
    x = concatenate([x, skip_connection_layer])
    
    '''fourth upsample'''
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    skip_connection_layer = base_model.get_layer('block2_conv2').output
    x = concatenate([x, skip_connection_layer])
    
    '''fifth upsample'''
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    skip_connection_layer = base_model.get_layer('block1_conv2').output
    x = concatenate([x, skip_connection_layer])
    
    
    #output layer
    output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax', padding='same')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

model = create_segmentation_model(input_shape=(224, 224, 3), num_classes=21)  # Assuming 21 classes including background
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

model.summary()
