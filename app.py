import streamlit as st
import numpy as np

from utils.extract_bottleneck_features import *
from utils.face_detector import face_detector
from utils.dog_detector import dog_detector
from utils.preprocess import path_to_tensor

from keras.models import load_model

from PIL import Image

with open('dogbreeds.txt', 'r') as f:
    dog_names = f.readlines()

dog_names = [x.strip() for x in dog_names]

model = load_model('saved_models/weights.best.VGG19.hdf5')


def VGG19_predict_breed(image):
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(image))
    # obtain predicted vector
    predicted_vector =  model.predict(bottleneck_feature)
    # return dog breed that is predcted by the model
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(file):
    if face_detector(file):
        dog_breed = model(file)
        return "The resembling dog breed of the human is {}".format(dog_breed)
    elif dog_detector(file):
        dog_breed = VGG19_predict_breed(file)
        return "The predicted breed of the dog is {}".format(dog_breed)
    else:
        return "Error: No dog or human face was detected in the image"

if __name__ == "__main__":

    file = st.file_uploader("Upload an image")
    
    if file is not None:

        img = Image.open(file)
        img = img.save('img.jpg')
        file_path = 'img.jpg'

        prediction = predict_dog_breed(file_path)
        st.write(prediction)
