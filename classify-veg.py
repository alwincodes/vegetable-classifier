import tensorflow as tf
import os

cnn = tf.keras.models.load_model("models/vegitable_predictor")
import numpy as np 
from keras.preprocessing import image

files = os.listdir("images_to_predict")
print("------------------ The Results ------------------")
for file in files:
    test_image = image.load_img(f"images_to_predict/{file}", target_size = (64, 64))

    test_image = image.img_to_array(test_image)
    test_image = test_image * 1/255 
    """
    the image array test_image is multiplied by 1/255 because during the training period we normalized the pixel data to be
    between 0 and 1 by multiplying the orginal pixel value with 1/255 (255 is the maximum pixel value for an image)
    """
    #adding extra dimensions because batch used for training was 32
    test_image = np.expand_dims(test_image, axis = 0)

    prediction = cnn.predict(test_image)[0]
    index = np.argmax(prediction) #argmax returns the array index with max values (since we used softmax as activation function the neuron with max value will be the prediction)
    classes = os.listdir("dataset/train")
    print(f"Prediction is {classes[index]} with {prediction[index]*100}% certainity")
    print(f"File name is {file} ")
    print("------------------")