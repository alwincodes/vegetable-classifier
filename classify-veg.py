import tensorflow as tf
import os

cnn = tf.keras.models.load_model("models/vegitable_predictor")
import numpy as np 
from keras.preprocessing import image

files = os.listdir("images_to_predict")
for file in files:
    test_image = image.load_img(f"images_to_predict/{file}", target_size = (64, 64))

    test_image = image.img_to_array(test_image)
    test_image = test_image * 1/255

    #adding extra dimensions because batch used for training was 32
    test_image = np.expand_dims(test_image, axis = 0)


    index = np.argmax(cnn.predict(test_image)[0])
    classes = os.listdir("dataset/train")
    print(f"Prediction is {classes[index]}")
    print(file)
    print("------------------")