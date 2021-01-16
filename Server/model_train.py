import cv2
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from PIL import Image
from random import choice
import os

IMAGE_SIZE = 256
IMAGE_DIRECTORY = '/mini project sudoku/image dataset'


def label_img(name):
    if name == 'sudoku':
        return np.array([0, 1])
    elif name == 'notsudoku':
        return np.array([1, 0])


def load_data():
    print("Loading images...")
    train_data = []
    directories = next(os.walk(IMAGE_DIRECTORY))[1]
    print(directories)

    for dirname in directories:
        print("Loading {0}".format(dirname))
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2]

        for i in range(120):
            try:
                image_name = choice(file_names)
                image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
                label = label_img(dirname)
                #print(label)
                img = Image.open(image_path)
                #print(img)
                img = img.convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                train_data.append([np.array(img), label])
            except:
                print("a")

    return train_data


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


training_data = load_data()
training_images = np.array([i[0] for i in training_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
training_labels = np.array([i[1] for i in training_data])

print('creating model')
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('training model')
model.fit(training_images, training_labels, batch_size=20, epochs=3, verbose=1)
model.save("sudoku_classification_model.h5")


'''
path = "image dataset/sudoku/image10.jpg"
img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
test_image = np.array(img).reshape(-1, IMAGE_SIZE, 1)
print("aaaaaaaaaaaaaaaaaaaaaa")
prediction = model.predict([[test_image]])

print(prediction)
'''

test_data = load_data()
test_images = np.array([i[0] for i in test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
test_labels = np.array([i[1] for i in test_data])

print('Loading model...')
model = load_model("sudoku_classification_model.h5")

print('Testing model...')
loss, acc = model.evaluate(test_images, test_labels, verbose=1)
print(loss)
print("accuracy: {0}".format(acc * 100))
