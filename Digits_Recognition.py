# Importing all the necessary modules

import cv2
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D
import matplotlib.pyplot as plot
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def model():
    # mnist.load_data() returns tuple of numpy arrays.
    # x_train, x_test = uint8 arrays of grayscale image data with shapes
    # x_test, y_test = uint8 arrays of digit labels(integers in range(0-9) with shapes
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = 255-x_train
    x_test = 255-x_test
    # train_test_split : splits the numpy arrays into random train and test subsets.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

    def preprocess(img):
        # cv2.equalizeHist takes grayscale image as input and gives histogram equalized image as output.
        img=cv2.equalizeHist(img)
        # The range of each RGB color is 0-255 so by dividing it by 255
        # 0-255 range can be described with 0-1 range
        # 0 ------> (0x00)
        # 1 ------> (0xFF)
        img = img/255.0
        return img
    # Reducing the ranges of x_train, x_test and x_val and bringing it in the range 0-1
    x_train = np.array(list(map(preprocess, x_train)))
    x_test = np.array(list(map(preprocess, x_test)))
    x_val = np.array(list(map(preprocess, x_val)))

    # Array of number of rows is reshaped to 28
    x_train = x_train.reshape(x_train.shape[0], 28,28,1)
    x_val = x_val.reshape(x_val.shape[0], 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28,28,1)

    # generating batches of tensor image data with real-time data augmentation.
    gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=25)
    gen.fit(x_train) # x_train is the training data through which the model is to be trained.

    # to_categorical() converts a numpy array having integers to a numpy array having binary values
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_val = to_categorical(y_val, 10)

    # Creating a sequential model
    model = Sequential()
    # Adding different layers in the model
    model.add(Conv2D(20, 5, padding="same", input_shape=(28, 28, 1),activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, 5, padding = "same", activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(250, activation='relu',use_bias=True))
    model.add(Dense(10, activation='softmax'))

    # Configuring the model for training.
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])

    BatchSize = 100
    TotalEpochs = 100
    # Allowing for data augmentation and data generators
    hist = model.fit_generator(gen.flow(x_train,y_train,BatchSize),steps_per_epoch=450,epochs=TotalEpochs,verbose=1,validation_data=(x_val, y_val))

    plot.figure(1)
    plot.plot(hist.history['loss'])
    plot.plot(hist.history['val_loss'])
    plot.legend(['train','validation'])
    plot.title('Loss')
    plot.figure(2)
    plot.plot(hist.history['accuracy'])
    plot.plot(hist.history['val_accuracy'])
    plot.legend(['train','validation'])
    plot.title('Accuracy')
    plot.show()

    score=model.evaluate(x_test,y_test,verbose=0)
    print("score = ",score[0])
    print("accuracy = ",score[1])
    model.save('t1.h5')


model()