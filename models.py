import keras
from keras.layers import LeakyReLU, Activation, Dense, Flatten
import random
from data_handeling import *
import numpy as np

IMAGE_DIM = (28,28)
INPUT_DIM = (28,28,2)

class Classifier:
    model = None
    m_type = None
    
    def __init__(self, m_type):
        self.m_type = m_type
        
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu',input_shape=INPUT_DIM))
        model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.20))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(72, activation='relu'))
        
        if m_type == "main":
            model.add(keras.layers.Dense(2, activation='softmax'))
            model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer='adagrad', loss='mean_squared_error', metrics=['accuracy'])
        
        self.model = model
    
    def classify(self, d):
        out = self.model.predict(np.reshape(d, (1, 28, 28, 2)))
        return out
        
    def train_model(self, data):
        print("Training: ", self.m_type)
        data.balance_data()
        (inputs, targets) = data.get_combined_data(self.m_type)
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            print("No data")
            return
        self.model.fit(inputs, targets, epochs=1, batch_size=32)
        
        

