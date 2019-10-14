import random
import numpy as np
from data_handeling import *
from models import *

TRAIN_ITER = 10
MAX_ARG_LENGTH = 10
IMAGE_DIM = (28,28)

class Ensemble:
    m = Classifier("main")
    c0 = Classifier("counter")
    c1 = Classifier("counter")

    def test_recursive(self, data_point, counter_channel, test_log, last_argument, arg_length):

        revaluation = 0
        
        (data_input, data_target) = data_point
        
        #print("Target: ", data_target)

        if arg_length != MAX_ARG_LENGTH:
            print("\t\t\t REVALUATION")
            revaluation = 1

        if arg_length <= 0:
            return (0.5, 0.5)  # If more that MAX arguments have been made, then the ensemble failed to converge to a solution
        else:
            arg_length -= 1

        counter_matrix = np.full(IMAGE_DIM, counter_channel)
        d = np.dstack([data_input, counter_matrix]) # combine image with counter channel
        test_log.add_data("m", data_target, d)      
        out = self.m.classify(d)[0]

        if out[0] > out[1]:         # M thinks the image is of type 0
            if data_target == label_0:
                test_log.data_m.n_correct += 1
            else:
                test_log.data_m.n_incorrect += 1

            #TODO: build this function through the logic of Experimental Design.
            # if not revaluation:
            #     if data_target == label_0:
            #         test_log.data_m.n_correct += 1
            #     else:
            #         test_log.data_m.n_incorrect += 1
            # else:   #TODO:  The following block of code should rollback the previous entries of test_log.data_m
            #     if last_argument == label_0: # M still thinks its label_0, but was prompted by C0
            #         if data_target == label_0: # C0 was WRONG
            #             test_log.data_m.n_correct += 0.5
            #             test_log.data_m.n_incorrect += 0.5
            #         else: # data_target is label_1, meaning C0 was RIGHT but M is WRONG.
            #             test_log.data_m.n_incorrect += 1



            print(f"\t\t argument FOR {label_0}", "(", data_target == label_0, ") -- Counter channel: ", counter_channel)
            
            if last_argument == label_0:
                print("\t\t\tsame argument twice")
                return (out[0], out[1])
                
            out_matrix = np.full(IMAGE_DIM, out[0])
            d_0 = np.dstack([data_input, out_matrix])     # What does this actually do ? Why dstack ?
            test_log.add_data("c0", data_target, d_0)
            out_c0 = self.c0.classify(d_0)[0][0]        # This number if > 0.5, argument for label_1 ?

            if out_c0 > 0.5:    # This says c0 does not agree with main. Counter argument is made.
                if data_target == label_0:
                    test_log.data_c0.n_incorrect += 1
                else:
                    test_log.data_c0.n_correct += 1
                print(f"\t\t\targument AGAINST {label_0}", "(", data_target == label_1, ")")
                return self.test_recursive((data_input, data_target), out_c0, test_log, label_0, arg_length)
            else:               # C0 does not make a counter argument.
                if data_target == label_1:
                    test_log.data_c0.n_incorrect += 1
                else:
                    test_log.data_c0.n_correct += 1
                print("\t\t\tno counter argument", "(", data_target == label_0, ")")
                return (out[0], out[1])
        else:                       # M thinks the image of type 1
            if data_target == label_1:
                test_log.data_m.n_correct += 1
            else:
                test_log.data_m.n_incorrect += 1
            print(f"\t\t argument FOR {label_1}", "(", data_target == label_1, ") -- Counter channel: ", counter_channel)
            
            if last_argument == label_1:
                print("\t\t\tsame argument twice")
                return (out[0], out[1])
                
            out_matrix = np.full(IMAGE_DIM, out[1])
            d_1 = np.dstack([data_input, out_matrix])       # add data_input and out_1
            test_log.add_data("c1", data_target, d_1)
            out_c1 = self.c1.classify(d_1)[0][0]

            if out_c1 > 0.5:    # This says c1 does not agree with main. Counter argument is made
                if data_target == label_1:
                    test_log.data_c1.n_incorrect += 1
                else:
                    test_log.data_c1.n_correct += 1
                print(f"\t\t\targument AGAINST {label_1}", "(", data_target == label_0, ")")
                return self.test_recursive((data_input, data_target), (1 - out_c1), test_log, label_1, arg_length)
            else:
                if data_target == label_0:
                    test_log.data_c1.n_incorrect += 1
                else:
                    test_log.data_c1.n_correct += 1
                print("\t\t\tno counter argument", "(", data_target == label_1, ")")
                return (out[0], out[1])

                # here we use (1-out_c1) so that a low counter_channel will be evidence against 1, and a high counter channel will be evidence against 0

    def test(self, inputs, targets, test_log):
        print("Let's test!\n")

        n_correct = 0
        n_wrong = 0
        
        for i in range(0, len(targets)):
            print("\tStart argument...\n")

            # Run the main classifier
            counter_channel = 0.5  #0.5 represents no counter argument made (yet)
            last_argument = 0.5    #0.5 represents no last argument 
            
            (out_0, out_1) = self.test_recursive((inputs[i], targets[i]), counter_channel, test_log, last_argument, MAX_ARG_LENGTH)
            
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
    (train_in, train_target, test_in, test_target) = load_data()
    
    print("Grab only part of the data")
    train_in = train_in[:10]
    train_target = train_target[:10]
    test_in = test_in[:100]
    test_target = test_target[:100]    

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
