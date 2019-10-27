import random
import numpy as np
from data_handeling import *
from models import *

TRAIN_ITER = 200
MAX_ARG_LENGTH = 10
IMAGE_DIM = (28,28)

class Test_Arguments:
    def __init__(self, data_point):
        self.data_point = data_point
        self.dialogue_channel = 0.5     #empty dialogue channel
        self.last_argument = -1        #no last argument
        self.argument_length = 0
        self.total_correct_m = 0
        self.total_correct_c0 = 0
        self.total_correct_c1 = 0

class Ensemble:
    m = Classifier("main")
    c0 = Classifier("counter_0")
    c1 = Classifier("counter_1")

    def test_recursive(self, test_args, test_log):
        
        (data_input, data_target) = test_args.data_point

        if test_args.argument_length >= MAX_ARG_LENGTH:
            return (0.5, 0.5)  # If more that MAX arguments have been made, then the ensemble failed to converge to a solution
        else:
            test_args.argument_length += 1

        dialogue_matrix = np.full(IMAGE_DIM, test_args.dialogue_channel)
        d = np.dstack([data_input, dialogue_matrix]) # combine image with dialogue channel
        test_log.add_data("m", data_target, d)
        out = self.m.classify(d)[0]
        
        if out[0] > out[1]:                         #M selected class 0
            label_chosen = label_0
            argument_strength = out[0]
            counter_generator = self.c0
            c_name = "c0"
        else:                                       #M selected class 1
            label_chosen = label_1
            argument_strength = out[1]
            counter_generator = self.c1
            c_name = "c1"
            
        if label_chosen == data_target:
            test_args.total_correct_m += 1
            
        #print(f"\t\t Argument for {label_chosen}", f" (Target = {data_target})", f" Dialogue channel: {test_args.dialogue_channel}")

        if test_args.last_argument == label_chosen:
            #print("\t\t\tsame argument twice")
            return (out[0], out[1])
                
        dialogue_matrix = np.full(IMAGE_DIM, argument_strength)     
        d = np.dstack([data_input, dialogue_matrix])
        test_log.add_data(c_name, data_target, d)
        out_c = counter_generator.classify(d)[0]
                
        if out_c <= 0.5: #no counter argument generated
            if label_chosen == data_target:
                test_log.log_accuracy(c_name, 1)    #log a correct counter argument
            else:
                test_log.log_accuracy(c_name, 0)    #log an incorrect counter argument
            #print("\t\t\tno counter argument", f"(Correct decision: {label_chosen == data_target})")
            return (out[0], out[1])
        else:           #counter argument generated
            if label_chosen == data_target:
                test_log.log_accuracy(c_name, 0)    #log an incorrect counter argument
            else:
                test_log.log_accuracy(c_name, 1)    #log a correct counter argument
            #print(f"\t\t\targument AGAINST {label_chosen}", f"(Correct decision: {label_chosen != data_target})")
            
            if c_name == "c0":
                test_args.dialogue_channel = out_c
            if c_name == "c1":
                test_args.dialogue_channel = (1-out_c)
            test_args.last_argument = label_chosen
            return self.test_recursive(test_args, test_log)

    def test(self, inputs, targets, test_log):
        print("Let's test!\n")

        n_correct = 0
        n_wrong = 0
        
        for i in range(0, len(targets)):
            #print("\tStart argument...\n")

            data_point = (inputs[i], targets[i])
            test_args = Test_Arguments(data_point)
            
            # Run the ensemble          
            (out_0, out_1) = self.test_recursive(test_args, test_log)
            
            #log accuracy of m
            m_accuracy = test_args.total_correct_m / test_args.argument_length
            test_log.log_accuracy("m", m_accuracy)

            if out_0 > out_1 and targets[i] == label_0:
                n_correct += 1
            elif out_1 > out_0 and targets[i] == label_1:
                n_correct += 1
            else:
                n_wrong += 1
                
        print("Accuracy Ensemble: ", n_correct/(n_correct + n_wrong))
        print("Accuracy Main: ", test_log.data_m.get_accuracy())
        print("Accuracy C0: ", test_log.data_c0.get_accuracy())
        print("Accuracy C1: ", test_log.data_c1.get_accuracy())
        
        test_log.reset_accuracies()

        return test_log

    def train(self, test_log):
        
        print("Let's train!\n")
        
        self.m.train_model(test_log.data_m)
        self.c0.train_model(test_log.data_c0)
        self.c1.train_model(test_log.data_c1)    
        
        test_log.clear_log()  

def main():
    (train_in, train_target, test_in, test_target) = load_data2()
    
    #print("Grab only part of the data")
    #train_in = train_in[:200]
    #train_target = train_target[:200]
    #test_in = test_in[:1000]
    #test_target = test_target[:1000]    
    
    print("Normalizing Data")
    train_in = image_normalize(train_in)   #make sure the values are between 0 and 1 (and not 0 and 255)
    test_in = image_normalize(test_in)
    
    ensemble = Ensemble()
    
    test_log = Test_Log()  # Here we will log all arguments made during the test run (for the purpose of training later)

    for i in range(0, TRAIN_ITER):
        # Run the ensemble, and record all arguments made
        ensemble.test(train_in, train_target, test_log)
        # the test_log contains all arguments made during testing

        # Train the classifiers on the new data
        ensemble.train(test_log)

    # Run the ensemble on test data
    print("Testing the test data")
    ensemble.test(test_in, test_target, test_log)


if __name__ == "__main__":
    main()
