import math
from PIL import Image, ImageOps
import os
from glob import glob
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf


images_files_path = 'path/to/query_image'


def get_image_from_query_path_eval(query_path, images_files_path= images_files_path):
    f = open(query_path, 'r')
    line = (f.readline())
    f.close()

    name, x1, y1, x2, y2 = line.split()
    name = name[5:] + '.jpg'

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1, y1, x2, y2 = int(math.floor(x1)), int(math.floor(y1)), int(math.floor(x2)), int(math.floor(y2))

    img_name = os.path.join(images_files_path, name)
    # print(img_name)
    image = Image.open(img_name)
    image = image.crop((x1, y1, x2, y2))

    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)

    return image

def get_image_from_query_path(img_name, images_files_path= images_files_path):
    image = Image.open(img_name)
    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    return image

delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
def extract_query_feature_DELF(image):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)

    return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))