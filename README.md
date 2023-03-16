# WhatsThatDog

The aim of this project is to predict the breed of a dog using transfer learning of pre-trained Convolutional Neural Networks (CNN) models, and build a web app using Streamlit to showcase the prediction results. 

The models are fine-tuned on a dataset of dog images to classify the images into different dog breeds.

The project provides a user-friendly web app built using Streamlit, where users can upload an image of a dog or a human face.

## Project Motivation

The motivation of this project is to leverage the power of transfer learning techniques and pre-trained CNN models to build an accurate and reliable dog breed prediction system. This project can be useful for dog breeders, trainers, and lovers, who want to identify the breed of a dog from an image.

## Repository Structure and File Description

```markdown
├── haarcascades
│   └── haarcascade_frontalface_alt.xml       # pre-trained human classifier
├── saved_models
│   └── weights.best.VGG19.hdf5               # trained CNN model 
├── utils
│   ├── dog_detector.py                       # python script to detect dogs in images
│   ├── extract_bottleneck_features.py        # bottleneck features for transfer learning
│   ├── face_detector.py                      # python script to detect faces in image
│   └── preprocess.py                         # python script to process data for model training
├── app.py                                    # Streamlit app
├── dog_app.ipynb                             # Model training shown in a jupyter notebook
├── dogbreeds.txt                             # List of dog breed names
├── README.md                                 # Readme file            

```

## Installations

To run this project, the following libraries and packages must be installed:
* Streamlit
* Pandas
* Keras
* Numpy
* Tensorflow
* CV2
* Tqdm
* Tempfile
* PIL

## How to Interact with the Project

1. Clone the repository to your local machine using the following command:
```
git clone https://github.com/zhikaichen99/WhatsThatDog.git
```

2. Run the `app.py` script to start the streamlit application by running the following command:
```
streamlit run app.y
```

3. A streamlit application will being running on `http://localhost:8501/`. It may take some time to set uo

4. Upload images on the streamlit application