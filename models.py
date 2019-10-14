import keras
from keras.layers import LeakyReLU, Activation, Dense, Flatten
from keras import optimizers
import random
from data_handeling import *
import numpy as np

INPUT_DIM = (28,28,2)

class Classifier:
    model = None
    m_type = None
    
    def __init__(self, m_type):
        self.m_type = m_type
        
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28, 2)))
        model.add(keras.layers.Dense(128, activation='relu', use_bias=True))
        
        if m_type == "main":
            model.add(keras.layers.Dense(2, activation='softmax', use_bias=True))
            #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0, nesterov=True)
            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))
            #sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0, nesterov=True)
            model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    def classify(self, d):
        out = self.model.predict(np.reshape(d, (1, 28, 28, 2)))
        #print("Output for ", self.m_type, ": ", out)
        return out
        
    def train_model(self, data):
        print("Training: ", self.m_type)
        data.balance_data()
        (inputs, targets) = data.get_combined_data(self.m_type)
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            print("No data")
            return
        self.model.fit(inputs, targets, epochs=3, batch_size=32)
        
        

