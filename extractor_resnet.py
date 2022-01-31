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

# DATAPATH    = r"C:\Users\PC\Desktop\CS336\data"
# model = get_extract_model()
# vectors =[]
# paths = []

# for path in tqdm(os.listdir(DATAPATH)):
#   img_path = os.path.join(DATAPATH, path)
#   img_vector = extract_vector(model, img_path)
#   vectors.append(img_vector)
#   paths.append(img_path)

# # print(img_path)
# vector_file = 'vector.pkl'
# path_file = 'path.pkl'
# pickle.dump(vectors, open(vector_file,'wb'))
# pickle.dump(paths, open(path_file,'wb'))
