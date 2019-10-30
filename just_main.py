import tensorflow as tf
import random
import numpy as np
import os
import cv2
import keras
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

INPUT_DIM = (28,28,1)

def load_data2():
    PATH = "./data/PetImagesStandard"
    
    cat_images = []
    for filename in os.listdir(PATH + "/Cat"):
        if filename.endswith('.jpg'):
            img = cv2.imread(PATH + "/Cat/" + filename, cv2.IMREAD_GRAYSCALE)
            cat_images.append(img)
    cat_labels = [0 for i in range(0, len(cat_images))]
    
    dog_images = []
    for filename in os.listdir(PATH + "/Dog"):
        if filename.endswith('.jpg'):
            img = cv2.imread(PATH + "/Dog/" + filename, cv2.IMREAD_GRAYSCALE)
            dog_images.append(img)
    dog_labels = [1 for i in range(0, len(dog_images))]
    
    images = cat_images + dog_images
    labels = cat_labels + dog_labels
    
    order = [i for i in range(0, len(images))]
    random.shuffle(order)
    
    shuffled_images = []
    shuffled_labels = []
    for i in order:
        shuffled_images.append(images[i])
        shuffled_labels.append(labels[i])
        
    n_train = int(0.8*len(shuffled_images))
    
    train_images = shuffled_images[:n_train]
    test_images = shuffled_images[n_train:]
    train_labels = shuffled_labels[:n_train]
    test_labels = shuffled_labels[n_train:]
    
    return train_images, train_labels, test_images, test_labels
    
def image_normalize(data):
    for i in range(0, len(data)):
        data[i] = np.true_divide(data[i], 255)
    return data
    
def main():
    print("Loading Data")
    (train_in, train_target, test_in, test_target) = load_data2()
    
    print("Format Labels")
    formated_train_target = []
    formated_test_target = []
    for t in train_target:
        if t == 0:
            formated_train_target.append([1, 0])
        else:
            formated_train_target.append([0, 1])
    for t in test_target:
        if t == 0:
            formated_test_target.append([1, 0])
        else:
            formated_test_target.append([0, 1])
    train_target = np.asarray(formated_train_target)
    test_target = np.asarray(formated_test_target)
    
    
    print("Normalizing Data")
    train_in = image_normalize(train_in)   #make sure the values are between 0 and 1 (and not 0 and 255)
    test_in = image_normalize(test_in)
    train_in = np.asarray(train_in)
    test_in = np.asarray(test_in)
    train_in = train_in[:, :, :, np.newaxis]
    test_in = test_in[:, :, :, np.newaxis]
    
    #build the model
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_DIM))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    
    print("Training...")
    model.fit(train_in, train_target, epochs=10, batch_size=32, verbose = 1)
    
    print("Testing...")
    score = model.evaluate(test_in, test_target, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    
    
if __name__ == "__main__":
    main()
    print("just_main v2, 1 epoch")
