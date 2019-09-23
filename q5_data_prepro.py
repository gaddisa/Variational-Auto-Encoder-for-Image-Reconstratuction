"""
############################################################################################
    code for data preprocessing and data augumentation
    @author: Gaddisa Olani
    Last updated: June 10,2019
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import cv2
import random
import pandas as pd
from scipy import ndimage

# Basic Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 47

# Prepare EMNISt data
def preprocess():
    # Get the data.
    nb_classes=NUM_LABELS
    train = pd.read_csv("hw_filtered_emnist-balanced-train.csv").values
    X_train = train[:, 1:]
    #separate the class label
    
    y_train = train[:, 0] # First data is label (already removed from X_train)

    #train_labels=y_train
    train_labels = onehot_encoding(y_train, nb_classes)
    train_data = normalize_and_reshape(X_train)
     
    return train_data, train_labels, train_data, train_labels
def onehot_encoding(data,nb_classes):
    """Extract the labels into a vector of int64 label IDs."""
    labels=data.astype(numpy.int64)
    return labels


"""
Data Augumentation 
   Shift, rotate, color change, gasuusian noise, image saturation,shearing,translation.....
"""

def normalize_and_reshape(data):
    
    num_images=data.shape[0]
    
    data=data.astype('float32')
    #data = data - (PIXEL_DEPTH / 2.0)
    #normalization
    data = data / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE)
    #data = numpy.reshape(data, [num_images, -1])
    print(data.shape)
    return data
def augument_my_training_data(images, labels):
    
    expanded_images = []
    expanded_labels = []

    for x, y in zip(images, labels):
           
        """
        Before starting the augumentation, first keep the original image in a list
        """
        expanded_images.append(x)
        expanded_labels.append(y)
        bg_value = numpy.median(x) 
        image = numpy.reshape(x, (-1, 28))
        
        
        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

   
    augumented_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(augumented_data)

    return augumented_data

def translation_image(image):
    x=random.randint(100,200)
    y=random.randint(100,200)
    rows, cols ,c= image.shape
    M = numpy.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def rotate(img):
    #randomly rotate the image #
    angle=random.randint(2,360)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, M, (w, h))
    return rotated_image
    

def saturate_image(image):
    saturation=random.randint(50,200)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = numpy.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def gausian_blur(image):
    blur=random.uniform(0, 1)
    image = cv2.GaussianBlur(image,(5,5),blur)
    return image

#this oepration involves dialation,erosion and substraction
def morphological_gradient_image(image):
    shift=random.randint(2,30)
    kernel = numpy.ones((shift, shift), numpy.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return image
#to add a random saturation jitter to an image. 

def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    return image

def contrast_image(image):
    contrast=random.randint(10,100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def original(x):
    return x

def sharpen_image(image):
    kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def edge_image(image,ksize):
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    return image

def flip(x):
    x=cv2.flip(x,1)
    return x
