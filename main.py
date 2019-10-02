import random
import numpy as np
from data_handeling import *
from models import *

TRAIN_ITER = 4
MAX_ARG_LENGTH = 10

class Ensemble:
    m = Classifier("main")
    c0 = Classifier("counter")
    c1 = Classifier("counter")

    def test_recursive(self, data_point, counter_channel, test_log, last_argument, arg_length):
        
        (data_input, data_target) = data_point

        if arg_length <= 0:
            return (0.5, 0.5)  # If more that MAX arguments have been made,
            # then the ensemble failed to converge to a solution
        else:
            arg_length -= 1

        counter_matrix = np.full(IMAGE_DIM, counter_channel)
        d = np.dstack([data_input, counter_matrix]) # combine image with counter channel
        test_log.add_data("m", data_target, d)      
        out = self.m.classify(d)[0]
        
        if out[0] > out[1]:
            if data_target == 0:
                test_log.data_m.n_correct += 1
            else:
                test_log.data_m.n_incorrect += 1
            print("\t\t argument FOR 0", "(", data_target == 0, ")")
            
            if last_argument == 0:
                print("\t\t\tsame argument twice")
                return (out[0], out[1])
                
            out_matrix = np.full(IMAGE_DIM, out[0])
            d_0 = np.dstack([data_input, out_matrix])     # add data_input and out_0
            test_log.add_data("c0", data_target, d_0)     #target is 1 if M is wrong (when the class is 1)
            out_c0 = self.c0.classify(d_0)[0]

            if out_c0 < 0.5:
                if data_target == 0:
                    test_log.data_c0.n_correct += 1
                else:
                    test_log.data_c0.n_incorrect += 1
                print("\t\t\tno counter argument", "(", data_target == 0, ")")
                return (out[0], out[1])
            else:
                if data_target == 1:
                    test_log.data_c0.n_correct += 1
                else:
                    test_log.data_c0.n_incorrect += 1
                print("\t\t\targument AGAINST 0", "(", data_target == 1, ")")
                return self.test_recursive((data_input, data_target), out_c0, test_log, 0, arg_length)
        else:
            if data_target == 1:
                test_log.data_m.n_correct += 1
            else:
                test_log.data_m.n_incorrect += 1
            print("\t\t argument FOR 1", "(", data_target == 1, ")")
            
            if last_argument == 1:
                print("\t\t\tsame argument twice")
                return (out[0], out[1])
                
            out_matrix = np.full(IMAGE_DIM, out[1])
            d_1 = np.dstack([data_input, out_matrix])       # add data_input and out_1
            test_log.add_data("c1", 1-data_target, d_1)     #target is 1 if M is wrong (when the class is 0)
            out_c1 = self.c1.classify(d_1)

            if out_c1 < 0.5:
                if data_target == 1:
                    test_log.data_c1.n_correct += 1
                else:
                    test_log.data_c1.n_incorrect += 1
                print("\t\t\tno counter argument", "(", data_target == 1, ")")
                return (out[0], out[1])
            else:
                if data_target == 0:
                    test_log.data_c1.n_correct += 1
                else:
                    test_log.data_c1.n_incorrect += 1
                print("\t\t\targument AGAINST 1", "(", data_target == 0, ")")
                return self.test_recursive((data_input, data_target), (1 - out_c1), test_log, 1, arg_length)
                # here we use (1-out_c1) so that a low counter_channel will be evidence against 1, and a high counter channel will be evidence against 0

    def test(self, inputs, targets):
        print("Let's test!\n")

        test_log = Test_Log()  # Here we will log all arguments made during the test run (for the purpose of training later)

        n_correct = 0
        n_wrong = 0
        
        for i in range(0, 4):#len(targets)):
            print("\tStart argument...\n")
            
            if i % 1000 == 0:
                print("Iteration: ", i)

            # Run the main classifier
            counter_channel = 0.5  #0.5 represents no counter argument made (yet)
            last_argument = 0.5    #0.5 represents no last argument 
            
            (out_0, out_1) = self.test_recursive((inputs[i], targets[i]), counter_channel, test_log, last_argument, MAX_ARG_LENGTH)
            
            if out_0 > out_1 and targets[i] == 0:
                n_correct += 1
            elif out_1 > out_0 and targets[i] == 1:
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
    (train_in, train_target, test_in, test_target) = load_data()
    
    ensemble = Ensemble()

    for i in range(0, TRAIN_ITER):
        # Run the ensemble, and record all arguments made
        test_log = ensemble.test(train_in, train_target)
        # the test_log contains all arguments made during testing

        # Train the classifiers on the new data
        ensemble.train(test_log)

    # Run the ensemble on test data
    ensemble.test(test_in, test_target)


if __name__ == "__main__":
    main()
