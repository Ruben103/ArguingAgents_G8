import random

TRAIN_ITER = 10
MAX_ARG_LENGTH = 10
DATA_SIZE = 10 #set this dynamically later

class Test_Log:
    data_m = []
    data_c0 = []
    data_c1 = []
    
def classify_c0(d): #implement later with a CNN
    a = random.uniform(0,1)
    return a
    
def classify_c1(d): #implement later with a CNN
    a = random.uniform(0,1)
    return a
    
def classify_m(d): #implement later with a CNN
    a = random.uniform(0,1)
    return (a, 1-a)

def test_recursive((data_input, data_target), counter_channel, test_log, arg_length):
    
    if arg_length <= 0:
        return (0.5, 0.5)   #If more that MAX arguments have been made, 
                            #then the ensemble failed to converge to a solution
    else:
        arg_length -= 1
    
    d = [data_input, counter_channel] #combine image with counter channel 
    (out_0, out_1) = classify_m(d)
    
    #########add d and target to test_log (data_m)
    
    if out_0 > out_1:
        print("\t\t argument FOR 0")
        
        d_0 = [data_input, out_0] #add data_input and out_0
        out_c0 = classify_c0(d_0)
        
        #########add d_0 and target to test_log (data_c0)
            #########here, target is 0 if M was correct, 1 if incorrect
        
        if out_c0 < 0.5:
            print("\t\t\tno counter argument")
            return (out_0, out_1)
        else:
            print("\t\t\targument AGAINST 0")
            return test_recursive((data_input, data_target), out_c0, test_log, arg_length)
    else:
        print("\t\t argument FOR 1")
        
        d_1 = [data_input, out_1] #add data_input and out_1
        out_c1 = classify_c1(d_1)
        
        #########add d_1 and target to test_log (data_c1)
            #########here, target is 0 if M was correct, 1 if incorrect
        
        if out_c1 < 0.5:
            print("\t\t\tno counter argument")
            
            return (out_0, out_1)
        else:
            print("\t\t\targument AGAINST 1")
            
            return test_recursive((data_input, data_target), (1-out_c1), test_log, arg_length)
                    #here we use (1-out_c1) so that a low counter_channel will be evidence against 1, and a high counter channel will be evidence against 0

def train(test_log):
    #Include this later
    print("Let's train!\n")
    
def test(data):
    print("Let's test!\n")
    
    test_log = Test_Log() #Here we will log all arguments made during the test run (for the purpose of training later)
    
    for i in range(0, DATA_SIZE):
        
        print("\tStart argument...\n")
    
        #Run the main classifier
        (out_0, out_1) = test_recursive((data[0][i], data[1][i]), 0.5, test_log, MAX_ARG_LENGTH)
                #0.5 represents no counter argument made (yet)
    
    return test_log

def main():
    train_data = [[i for i in range(0,DATA_SIZE)],[j%2 for j in range(0,DATA_SIZE)]]
    test_data = [[i for i in range(0,DATA_SIZE)],[j%2 for j in range(0,DATA_SIZE)]]

    for i in range(0,TRAIN_ITER):
        #Run the ensemble, and record all arguments made
        test_log = test(train_data) 
                #the test_log contains all arguments made during testing
    
        #Train the classifiers on the new data
        train(test_log)

    #Run the ensemble on test data
    test(test_data)

if __name__ == "__main__":
    main()
