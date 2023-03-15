#from keras.preprocessing.image import load_image, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
from tqdm import tqdm

import numpy as np

def path_to_tensor(img_path):
    """
    Loads an image from a given file path and returns a 4D tensor

    Inputs:
        img_path (str): file path of the image file
    Outputs:
        numpy.ndarray: A 4D tensor of shape (1,224,224,3) representing the loaded image
    """
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """
    Loads a list of images from a list of paths and returns them as a 4D tensor

    Inputs:
        img_paths (str): file paths of the image files
    Outputs:
        numpy.ndarray: A 4D tensor of shape (1,224,224,3) representing the loaded image
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)