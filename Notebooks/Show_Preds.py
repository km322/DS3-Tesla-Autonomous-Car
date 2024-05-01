import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import cv2
import os
import torchvision.transforms as transforms
from PIL import Image 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
from keras.applications import vgg16
from keras.models import Model,load_model
import keras
from keras.layers import Input, Conv2D, Conv2DTranspose,AveragePooling2D, MaxPooling2D,UpSampling2D,LeakyReLU, concatenate, Dropout,BatchNormalization,Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from keras.models import load_model
model = load_model('/Users/derke/Documents/GitHub/DS3-Tesla-Autonomous-Car/finished-test.keras')
current_dir = '/Users/derke/Desktop'
def store_data(data):
    RGB_files = sorted(os.listdir(current_dir + "/archive-2/"+data+"/"+data+"/CameraRGB"))
    Seg_files = sorted(os.listdir(current_dir + "/archive-2/"+data+"/"+data+"/CameraSeg"))
    for i in range(len(RGB_files)):
        RGB_files[i] = current_dir + "/archive-2/"+data+"/"+data+"/CameraRGB/" + RGB_files[i]
    for i in range(len(Seg_files)):
        Seg_files[i] = current_dir + "/archive-2/"+data+"/"+data+"/CameraSeg/" + Seg_files[i]

    return (RGB_files, Seg_files)

image_files = []
mask_files = []
image_files += store_data("dataA")[0]
image_files += store_data("dataB")[0]
image_files += store_data("dataC")[0]
image_files += store_data("dataD")[0]
image_files += store_data("dataE")[0]

mask_files += store_data("dataA")[1]
mask_files += store_data("dataB")[1]
mask_files += store_data("dataC")[1]
mask_files += store_data("dataD")[1]
mask_files += store_data("dataE")[1]

        

def preprocess_input_image(image_path, target_size=(224, 224)):
    # Load the image
    image = load_img(image_path, target_size=target_size)
    # Convert the image to a numpy array
    image_array = img_to_array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
            
def predict_mask(model, image_array):
    # Predict the mask
    prediction = model.predict(image_array)
    return prediction
            
            
def postprocess_prediction(prediction, target_size=(224, 224)):
    # Take the argmax across channels to convert to class indices
    predicted_mask = np.argmax(prediction, axis=-1)
    # Remove the batch dimension
    predicted_mask = predicted_mask[2]
    return predicted_mask



def display(index, target_size=(224, 224)):
    img_path = image_files[index]
    mask_path = mask_files[index]
    image_array = preprocess_input_image(img_path)
    prediction = predict_mask(model, image_array)
    predicted_mask = postprocess_prediction(prediction)
    
    # Load and display the original image
    original_image = load_img(img_path, target_size=target_size)
    
    # Load and display the true mask
    # Ensure this is adapted if your masks are not single-channel grayscale images
    true_mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')
    true_mask = np.squeeze(np.array(true_mask))  # Remove if unnecessary
    
    # Display the predicted mask
    # Ensure predicted_mask is correctly postprocessed to be displayable
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')  # Turn off axis
    
    ax[1].imshow(true_mask, cmap='jet')  # Use appropriate colormap
    ax[1].set_title("True Mask")
    ax[1].axis('off')  # Turn off axis
    
    ax[2].imshow(predicted_mask, cmap='jet')  # Use appropriate colormap
    ax[2].set_title("Predicted Mask")
    ax[2].axis('off')  # Turn off axis
    
    plt.show()

display(0)


