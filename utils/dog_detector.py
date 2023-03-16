from keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from utils.preprocess import path_to_tensor, paths_to_tensor
import numpy as np


# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    """
    Returns the prediction vector for an image located at img_path

    Inputs:
        img_path (str): file path of the image file
    Outputs:
        Numpy array: The prediction vector for the image
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    """
    Return True if a dog is detected in the image stored at img_path

    Inputs:
        img_path (str): file path of the image file
    Ouputs:
        bool: True if a dog is detected. False otherwise
    """
    prediction = ResNet50_predict_labels(img_path)
    # dog breeds are classified between index 151 and 268, hence the condition
    return ((prediction <= 268) & (prediction >= 151)) 