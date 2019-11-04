import keras
from keras.layers import LeakyReLU, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
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
        # Shared CNN structure for both main and counter-argument networks
        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_DIM))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        
        if m_type == "main":
            # Add 2 outputs for the main classifier
            model.add(keras.layers.Dense(2, activation='softmax'))
            #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0, nesterov=True)
            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            # Add 1 output for Counter-Argument generator
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            #sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0, nesterov=True)
            model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
    
    def classify(self, d):
        """
        Return the prediction labels for a given datapoint.
        """
        out = self.model.predict(np.reshape(d, (1, 28, 28, 2)))
        #print("Output for ", self.m_type, ": ", out)
        return out
        
    def train_model(self, data):
        """
        Define a training strategy for the models.
        """
        print("Training: ", self.m_type)
        data.balance_data()
        (inputs, targets) = data.get_combined_data(self.m_type)
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            print("No data")
            return
        self.model.fit(inputs, targets, epochs=1, batch_size=32)
        
        

