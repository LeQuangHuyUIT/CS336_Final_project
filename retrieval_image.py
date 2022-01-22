from extractor_query import *
import pickle
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform


def get_resized_db_image_paths(destfolder):
    return sorted(list(glob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))

'''
load database features
'''
name_locations_agg = "locations_agg.npy"
name_descriptors_agg = "descriptors.npy"
name_accumulated = "accumulated_indexes_boundaries.pkl"

# remember to change image file path to folder contain image
images_files_path = '/content/drive/MyDrive/CS336_Image_Retrieval/oxbuild_images'


# output_path: path to folder feature 
output_path = "/content/drive/MyDrive/CS336_Image_Retrieval"
locations_save_path = os.path.join(output_path, name_locations_agg) 
descriptors_save_path = os.path.join(output_path, name_descriptors_agg)
accumulated_indexes_boundaries_save_path = os.path.join(output_path, name_accumulated)


'''
load features
'''
db_images = get_resized_db_image_paths(images_files_path)
locations_agg = np.load(locations_save_path)
descriptors_agg = np.load(descriptors_save_path)

open_file = open(accumulated_indexes_boundaries_save_path, "rb")
accumulated_indexes_boundaries = pickle.load(open_file)
open_file.close()


'''
build kdTree
'''
dtree = cKDTree(descriptors_agg)

# function for retrieval
def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start,accumulated_index_end)

def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries,\
                        query_image_locations, locations_agg):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(query_image_locations[i])
                locations_2_use_db.append(locations_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)


def retrieval_image(image_path, top_result):
    query_image = get_image_from_query_path(image_path)
    query_features = extract_query_feature_DELF(query_image)
    query_image_locations = query_features['locations']
    query_image_descriptors = query_features['descriptors']

    distance_threshold = 0.8
    # K nearest neighbors
    K = 8
    distances, indices = dtree.query(query_image_descriptors, distance_upper_bound=distance_threshold, k = K, n_jobs=-1)

    # Find the list of unique accumulated/aggregated indexes
    unique_indices = np.array(list(set(indices.flatten())))

    unique_indices.sort()
    if unique_indices[-1] == descriptors_agg.shape[0]:
        unique_indices = unique_indices[:-1]
    unique_image_indexes = np.array(
        list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) 
                for index in unique_indices])))
    
    inliers_counts = []
    for index in unique_image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index, \
                                                                        indices,\
                                                                        accumulated_indexes_boundaries,\
                                                                        query_image_locations,\
                                                                        locations_agg)
        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
                (locations_2_use_db, locations_2_use_query), # source and destination coordinates
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=100)
        except:
            inliers = None
        # If no inlier is found for a database candidate image, we continue on to the next one.
        if inliers is None or len(inliers) == 0:
            continue
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})

    top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)

    results = [db_images[item['index']] for i, item in enumerate(top_match[:top_result])]
    return results