from extractor_query import *
from PIL import Image, ImageOps

img_name = 'index.jpg'
image = Image.open(img_name)

image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)

features = extract_query_feature_DELF(image)
print(features)