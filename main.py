import random
import numpy as np
from unpack_data import *

TRAIN_ITER = 2
MAX_ARG_LENGTH = 10
IMAGE_DIM = (28,28)

class Data:         #these are separated in order to later check to make sure the data is balanced
    target_0 = []
    target_1 = []

class Test_Log:
    data_m = Data()
    data_c0 = Data()
    data_c1 = Data()


def classify_c0(d):  # implement later with a CNN
    a = random.uniform(0, 1)
    return a


def classify_c1(d):  # implement later with a CNN
    a = random.uniform(0, 1)
    return a


def classify_m(d):  # implement later with a CNN
    a = random.uniform(0, 1)
    return (a, 1 - a)


def test_recursive(data_point, counter_channel, test_log, last_argument, arg_length):
    (data_input, data_target) = data_point

    if arg_length <= 0:
        return (0.5, 0.5)  # If more that MAX arguments have been made,
        # then the ensemble failed to converge to a solution
    else:
        arg_length -= 1

    counter_matrix = np.full(IMAGE_DIM, counter_channel)

    d = np.dstack([data_input, counter_matrix]) # combine image with counter channel

    if data_target == 1:
        test_log.data_m.target_1.append(d)
    else:
        test_log.data_m.target_0.append(d)
        
    (out_0, out_1) = classify_m(d)

    if out_0 > out_1:
        print("\t\t argument FOR 0")
        
        if last_argument == 0:
            print("\t\t\tsame argument twice")
            return (out_0, out_1)
            
        out_matrix = np.full(IMAGE_DIM, out_0)
            
        d_0 = np.dstack([data_input, out_matrix])  # add data_input and out_0

        if data_target == 1: #M is wrong
            test_log.data_m.target_1.append(d_0) #counter
        else:
            test_log.data_m.target_0.append(d_0) #no counter
            
        out_c0 = classify_c0(d_0)

        if out_c0 < 0.5:
            print("\t\t\tno counter argument")
            return (out_0, out_1)
        else:
            print("\t\t\targument AGAINST 0")
            return test_recursive((data_input, data_target), out_c0, test_log, 0, arg_length)
    else:
        print("\t\t argument FOR 1")
        
        if last_argument == 1:
            print("\t\t\tsame argument twice")
            return (out_0, out_1)
            
        out_matrix = np.full(IMAGE_DIM, out_1)

        d_1 = np.dstack([data_input, out_matrix])  # add data_input and out_1

        if data_target == 1: #M is right
            test_log.data_m.target_0.append(d_1) #no counter
        else:
            test_log.data_m.target_1.append(d_1) #counter
            
        out_c1 = classify_c1(d_1)

        if out_c1 < 0.5:
            print("\t\t\tno counter argument")
            return (out_0, out_1)
        else:
            print("\t\t\targument AGAINST 1")
            return test_recursive((data_input, data_target), (1 - out_c1), test_log, 1, arg_length)
            # here we use (1-out_c1) so that a low counter_channel will be evidence against 1, and a high counter channel will be evidence against 0


def train(test_log):
    
    #make sure that the data used for training is even (same number for each class)
    
    # Include this later
    print("Let's train!\n")
    


def test(inputs, targets):
    print("Let's test!\n")

    test_log = Test_Log()  # Here we will log all arguments made during the test run (for the purpose of training later)

    for i in range(0, len(targets)):
        print("\tStart argument...\n")

        # Run the main classifier
        counter_channel = 0.5  #0.5 represents no counter argument made (yet)
        last_argument = 0.5    #0.5 represents no last argument 
        
        (out_0, out_1) = test_recursive((inputs[i], targets[i]), counter_channel, test_log, last_argument, MAX_ARG_LENGTH)

    return test_log


def main():
    
    (train_in, train_target, test_in, test_target) = load_data()


    for i in range(0, TRAIN_ITER):
        # Run the ensemble, and record all arguments made
        test_log = test(train_in, train_target)
        # the test_log contains all arguments made during testing

        # Train the classifiers on the new data
        train(test_log)

    # Run the ensemble on test data
    #test(test_data)


if __name__ == "__main__":
    main()
