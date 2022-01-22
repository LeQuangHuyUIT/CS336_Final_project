from tqdm import tqdm 
import os
# import tensorflow as tf 
from keras.applications.resnet import ResNet101,preprocess_input
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image

from PIL import Image,ImageOps
import pickle
import numpy as np
import time
import cv2

def get_extract_model():
  resnet101_model= ResNet101(weights="imagenet")
  extract_model = Model( inputs=resnet101_model.input, outputs = resnet101_model.layers[-2].output)
  return extract_model

def processing_img(img):
  img = img.resize((224,224))
  img = img.convert('RGB')
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x

def extract_vector(model ,path):
  img =Image.open(path)
#   img =get_image_from_query_path(path)
  img_tensor = processing_img(img)
  vector = model.predict(img_tensor)[0]
  vector = vector / np.linalg.norm(vector)
  return vector