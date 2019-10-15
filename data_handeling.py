import tensorflow as tf
import random
import numpy as np

label_0 = 3
label_1 = 8

class Data:         #these are separated in order to later check to make sure the data is balanced
    
    def __init__(self):
        self.target_0 = []
        self.target_1 = []
        self.n_correct = 0 # 8
        self.n_incorrect = 0 # 9
        self.prev_correct = 0 # 8 8
        self.prev_incorrect = 0 # 9 9
    
    def get_accuracy(self):
        if self.n_correct + self.n_incorrect > 0:
            return self.n_correct / (self.n_correct + self.n_incorrect)
        else:
            return None

    def set_previous(self):
        self.prev_correct = self.n_correct
        self.prev_incorrect = self.n_incorrect

    def rollback(self):
        self.n_correct = self.prev_correct
        self.n_incorrect = self.prev_incorrect
    
    def get_combined_data(self, m_type):
        if m_type == "main":
            combined_data = self.target_0.copy()
            combined_targets = [[1, 0] for i in range(0, len(self.target_0))]
            combined_data.extend(self.target_1.copy())
            combined_targets.extend([[0, 1] for i in range(0, len(self.target_1))])
        else:
            combined_data = self.target_0.copy()
            combined_targets = [0 for i in range(0, len(self.target_0))]
            combined_data.extend(self.target_1.copy())
            combined_targets.extend([1 for i in range(0, len(self.target_1))])
        return(np.asarray(combined_data), np.asarray(combined_targets))
    
    def balance_data(self):
        print("Balancing: ", len(self.target_0), " : ", len(self.target_1))
        if len(self.target_0) == 0 or len(self.target_1) == 0:
            print("Error, cannot balance data")
            self.target_0 = []
            self.target_1 = []
            return
        
        n = max(len(self.target_0), len(self.target_1))
        balanced_0 = []
        balanced_1 = []
        while len(balanced_0) < n:
            l = self.target_0.copy()
            random.shuffle(l)
            balanced_0.extend(l)
        while len(balanced_1) < n:
            l = self.target_1.copy()
            random.shuffle(l)
            balanced_1.extend(l)
        self.target_0 = balanced_0[:n]
        self.target_1 = balanced_1[:n]
        
    def clear_data(self):
        self.target_0 = []
        self.target_1 = []

class Test_Log:
    
    def __init__(self):
        self.data_m  = Data()
        self.data_c0 = Data()
        self.data_c1 = Data()
        self.data_ens= Data()
    
    def add_data(self, model, target, d):
        if model == "m":
            if target == label_0:
                self.data_m.target_0.append(d)
            elif target == label_1:
                self.data_m.target_1.append(d)
            else:
                print("Error adding data")
                quit()
        elif model == "c0":
            if target == label_0:       #if the target is 0, M is right 
                self.data_c0.target_0.append(d)
            elif target == label_1:
                self.data_c0.target_1.append(d)
            else:
                print("Error adding data")
                quit()
        elif model == "c1":
            if target == label_1:       #if the target is 1, M is right
                self.data_c1.target_0.append(d)
            elif target == label_0:
                self.data_c1.target_1.append(d)
            else:
                print("Error adding data")
                quit()
        else:
            print("Error adding data :(")
            quit()
    
    def clear_log(self):
        self.data_m.clear_data()
        self.data_c0.clear_data()
        self.data_c1.clear_data()
        self.data_ens.clear_data()
        
    def reset_accuracies(self):
        self.data_m.n_correct = 0
        self.data_m.n_incorrect = 0
        self.data_c0.n_correct = 0
        self.data_c0.n_incorrect = 0
        self.data_c1.n_correct = 0
        self.data_c1.n_incorrect = 0
        self.data_ens.n_correct = 0
        self.data_ens.n_incorrect = 0


def load_data():
    filtered_train_images = []
    filtered_train_labels = []
    filtered_test_images = []
    filtered_test_labels = []
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_ctr_0 = 0
    test_ctr_0 = 0
    train_ctr_1 = 0
    test_ctr_1 = 0
    for train_idx in range(0, len(train_labels)):
        if (train_labels[train_idx] == label_0):
            filtered_train_images.append(train_images[train_idx])
            filtered_train_labels.append(train_labels[train_idx])
            train_ctr_0 += 1
        if (train_labels[train_idx] == label_1):
            filtered_train_images.append(train_images[train_idx])
            filtered_train_labels.append(train_labels[train_idx])
            train_ctr_1 += 1
    print(f"Number of train images:\n {label_0}: {train_ctr_0}\t {label_1}: {train_ctr_1}")

    for test_idx in range(0, len(test_labels)):
        if (test_labels[test_idx] == label_0):
            filtered_test_images.append(test_images[test_idx])
            filtered_test_labels.append(test_labels[test_idx])
            test_ctr_0 += 1
        if (test_labels[test_idx] == label_1):
            filtered_test_images.append(test_images[test_idx])
            filtered_test_labels.append(test_labels[test_idx])
            test_ctr_1 += 1
    print(f"Number of test images:\n {label_0}: {test_ctr_0}\t {label_1}: {test_ctr_1}")
    # TODO: Add a try catch to check if data is always returned with expected shapes and sizes
    return filtered_train_images, filtered_train_labels, filtered_test_images, filtered_test_labels
    
def image_normalize(data):
    for i in range(0, len(data)):
        data[i] = np.true_divide(data[i], 255)
    return data
