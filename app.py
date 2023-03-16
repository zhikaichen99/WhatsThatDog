import streamlit as st
import numpy as np
import tempfile

from utils.extract_bottleneck_features import *
from utils.face_detector import face_detector
from utils.dog_detector import dog_detector
from utils.preprocess import path_to_tensor

from keras.models import load_model

from PIL import Image


# get a list of dog breeds
with open('dogbreeds.txt', 'r') as f:
    dog_names = f.readlines()

dog_names = [x.strip() for x in dog_names]

# load the pre-trained VGG19 model from the saved model folder
model = load_model('saved_models/weights.best.VGG19.hdf5')


def VGG19_predict_breed(image):
    """
    Returns the predicted breed of a dog in the image

    Inputs:
        image (str): file path of the image file
    Outputs:
        str: The predicted breed of the dog in the image
    """
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(image).astype('float32')/255)
    # obtain predicted vector
    predicted_vector =  model.predict(bottleneck_feature)
    # return dog breed that is predcted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(file):
    """
    Returns a string describing the predicted dog breed or an error message

    Inputs:
        file (str): file path of the image file
    Outputs:
        str: The predicted dog breed or an error message if no dog or human face detected
    """
    if face_detector(file):
        dog_breed = VGG19_predict_breed(file)
        return "The resembling dog breed of the human is {}".format(dog_breed)
    elif dog_detector(file):
        dog_breed = VGG19_predict_breed(file)
        return "The predicted breed of the dog is {}".format(dog_breed)
    else:
        return "Error: No dog or human face was detected in the image"



if __name__ == "__main__":

    file = st.file_uploader("Upload an image")
    
    if file is not None:
        # save the uploaded file to a temporary file on disk
        with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
            tmp_file.write(file.read())
            file_path = tmp_file.name
        print("HERE")
        print(file_path)

        # display the uploaded image on the streamlit app
        st.image(file, use_column_width = True)

        # display a header indicating the prediction is being loaded
        prediction_header = st.empty()
        prediction_header.header("Loading Prediction")
 
        # predict the dog breed from the uploaded image and display the results
        prediction = predict_dog_breed(file_path)
        prediction_header.header(prediction)
