import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.__version__

#train set
train_datagen = ImageDataGenerator(
                    rescale = 1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True
                    )

training_set = train_datagen.flow_from_directory(
                    "dataset/train",
                    target_size = (64, 64),
                    batch_size = 32,
                    class_mode = "categorical"
                    )

#test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
                    "dataset/test",
                    target_size = (64, 64),
                    batch_size = 32,
                    class_mode = "categorical"
                    )

# #validate set
# validate_datagen = ImageDataGenerator(rescal = 1./255)
# validate_set = validate_datagen.flow_from_directory(
#                     "dataset/validation",
#                     target_size = (64, 64),
#                     batch_size = 32,
#                     class_mode = "categorical"
#                     )


#intitializing
cnn = tf.keras.models.Sequential()
#convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =  3, activation= "relu", input_shape = (64, 64, 3)))
#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
#second convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size =  3, activation= "relu"))
#second pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units = 512, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 128, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 64, activation="relu"))
cnn.add(tf.keras.layers.Dense(units = 64, activation="relu"))

cnn.add(tf.keras.layers.Dense(units = 15, activation="softmax"))
#softmax because we are doing categorical classification

#compiling the cnn
cnn.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
#training the cnn on train set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25 )

cnn.save("models/vegitable_predictor")


# cnn = tf.keras.models.load_model("models/cat_dog")
# import numpy as np 
# from keras.preprocessing import image

# files = os.listdir("dataset/single_prediction")
# for file in files:
#     test_image = image.load_img(f"dataset/single_prediction/{file}", target_size = (64, 64))

#     test_image = image.img_to_array(test_image)

#     #adding extra dimensions because batch used for training was 32
#     test_image = np.expand_dims(test_image, axis = 0)

#     if(cnn.predict(test_image)[0][0] == 1):
#         print("dog")
#     else:
#         print("cat")

#     print(cnn.predict(test_image)[0][0])
#     print(file)
#     print("------------------")

