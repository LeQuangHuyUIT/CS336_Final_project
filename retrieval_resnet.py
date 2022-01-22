from operator import imod
from extractor_resnet import *
import math
from scipy import spatial
from glob import glob

DATAPATH    = "/content/drive/MyDrive/CS336/data"
gt_files_path = '/content/drive/MyDrive/CS336/gt_files'
load_vectors = pickle.load(open('/content/drive/MyDrive/CS336/vector.pkl','rb'))
load_paths = pickle.load(open('/content/drive/MyDrive/CS336/path.pkl','rb'))

query_files = sorted(glob(gt_files_path + '/*query.txt'))

def get_image_from_query_path(query_path, images_files_path= DATAPATH):
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

def compute_AP(relevant_set, results):
    '''
        results : list of image name return from system
        relevant_set: list of image name from _good and _ok files
    '''
    total_correct = 0
    total = 0
    precisions = []

    for name in results:
        total += 1
        if name in relevant_set:
            total_correct += 1
            precisions.append(total_correct / total)
    
    if len(precisions) == 0:
        return 0

    return sum(precisions) / len(precisions)

def get_ground_truth(query_path):
    query_basename = os.path.basename(query_path)
    dirname = os.path.dirname(query_path)
    basename = query_basename.rstrip("_query.txt")

    good_file = basename + '_good.txt'
    ok_file = basename + '_ok.txt'
    junk_file = basename + '_junk.txt'

    good_file_path = os.path.join(dirname, good_file)
    ok_file_path = os.path.join(dirname, ok_file)

    ground_truth = []

    with open(good_file_path, 'r') as f:
        while True:
            line = f.readline()
            if  len(line) == 0:
                break
            line = line.rstrip('\n')
            ground_truth.append(line)

    with open(ok_file_path, 'r') as f:
        while True:
            line = f.readline()
            if  len(line) == 0:
                break
            line = line.rstrip('\n')
            ground_truth.append(line)

    return ground_truth

def get_retrieved_file(results_from_retrival):
    return [os.path.basename(x).strip(".jpg") for x in results_from_retrival]

def retrieval(query_path):
  # Khoi tao model
  model = get_extract_model()

  # Trich dac trung anh search
  search_vector = extract_vector(model, query_path)

  # Tinh khoang cach tu search_vector den tat ca cac vector
  
  # distance = np.linalg.norm(load_vectors - search_vector, axis=1)
  distance = np.apply_along_axis(spatial.distance.cityblock, 1,load_vectors,search_vector)
  # Sap xep va lay ra K vector co khoang cach ngan nhat
  K = 16
  ids = np.argsort(distance)[:K]

  # Tao oputput
#   nearest_image = [(load_paths[id], distance[id]) for id in ids]
  nearest_image_path = [(load_paths[id]) for id in ids]
  return nearest_image_path
# ----------------------------------------------
# gt = get_ground_truth(query_files[0])
# image_retrieved_path = retrieval(query_files[0])
# results = get_retrieved_file(image_retrieved_path)
# print(results)
# print(gt)
# print(compute_AP(gt, results))

# s = 0
# for i in range(0,len(query_files)):
#   gt = get_ground_truth(query_files[i])
#   image_retrieved_path = retrieval(query_files[i])
#   results = get_retrieved_file(image_retrieved_path)
#   AP = compute_AP(gt, results)
#   s = s + AP
# print("mAP: ",s / len(query_files)) 