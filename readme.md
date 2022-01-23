# Vegetable-classifier ML model

This project is my first attempt at creating a convolutional neural
network, Through this project I trained a model which then can be
used to categorize Vegetables to 15 categories just by inputing its
image.

### Categories of Vegitables the model can classify

('Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato')

## Libraries used

-   Tensorflow
-   numpy
-   pillow (PIL)

### Getting started

requirement.txt file is included which can be used with a virtual environment get started with this project

1. create a virtual environment (if you are new to venv a google search will resolve your issues)
2. enter into the virtual environment
3. install the necessary packages (you can use the requirement.txt file)
4. get the dataset which can be found [here](https://www.kaggle.com/misrakahmed/vegetable-image-dataset)
5. now train the model by running model-creator.py (this will take some time)
6. now place the images that you need to classify in images_to_predict directory
7. run classify-veg.py you will now get a prediction from the model

## Screenshots

The directory strucuture of this application
![App Screenshot](https://github.com/alwincodes/vegetable-classifier/blob/main/screen-shots/directory%20structure.png?raw=true)

First set of prediction done using this application
![App Screenshot](https://github.com/alwincodes/vegetable-classifier/blob/main/screen-shots/first%20set%20of%20predictions%20using%20new%20data.png?raw=true)
