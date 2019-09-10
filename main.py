TRAIN_ITER = 10
MAX_ARG_LENGTH = 10

class Test_Log:
    data_m = []
    data_c0 = []
    data_c1 = []

def test_recursive(data_point, counter_channel, test_log, arg_length):
    
    if arg_length <= 0:
        return (0.5, 0.5)
    else:
        arg_length -= 1
    
    d = #add data_point and counter_channel
    (out_0, out_1) = classify_m(d)
    
    #########add d and target to test_log (data_m)
    
    if out_0 > out_1:
        d_0 = #add data_point and out_0
        out_c0 = classify_c0(d_0)
        
        #########add d_0 and target to test_log (data_c0)
            #########here, target is 0 if M was correct, 1 if incorrect
        
        if out_c0 < 0.5:
            return (out_0, out_1)
        else:
            return test_recursive(data_point, out_c0, test_log, arg_length)
    else:
        d_1 = #add data_point and out_1
        out_c1 = classify_c1(d_1)
        
        #########add d_1 and target to test_log (data_c1)
            #########here, target is 0 if M was correct, 1 if incorrect
        
        if out_c1 < 0.5:
            return (out_0, out_1)
        else:
            return test_recursive(data_point, (1-out_c1), test_log, arg_length)
                    #here we use (1-out_c1) so that a low counter_channel will be evidence against 1, and a high counter channel will be evidence against 0

def train(data_m, data_c0, data_c1):
    #Train M
    #Train C0
    #Train C1
    
def test(input_data):
    test_log = Test_Log()
    
    for i in range(0, #size_data):
    
        #Run the main classifier
        (out_0, out_1) = test_recursive(data[i], 0.5, test_log, MAX_ARG_LENGTH)
                #0.5 represents no counter argument made
    
    return #the test_log

def main():
    train_data = #The actual data from the dataset we use

    for i in range(0,TRAIN_ITER):
        #Run the ensemble, and record all arguments made
        (data_m, data_c0, data_c1) = test(train_data) 
                #Here data_x referes to all the arguments which 'x' made during testing, and whether they were correct
    
        #Train the classifiers on the new data
        train(data_m, data_c0, data_c1)

    #Run the ensemble on test data
    test(test_data)

if __name__ == "__main__":
    main()
