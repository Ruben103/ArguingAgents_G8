import keras



def gen_main_model(input_shape=(28, 28, 2), num_classes=2):
    main_model = keras.Sequential()
    main_model.add(keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
    main_model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    main_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    main_model.add(keras.layers.Dropout(0.20))
    main_model.add(keras.layers.Flatten())
    main_model.add(keras.layers.Dense(72, activation='relu'))
    main_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # TODO: Add model compilation.
    # main_model.compile(loss=keras.losses.categorical_crossentropy,              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return main_model

def gen_counter_arg_model(input_shape=(28, 28, 2), num_classes=1):
    counter_arg_model = keras.Sequential()
    counter_arg_model.add(keras.layers.Conv2D(20, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
    counter_arg_model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
    counter_arg_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    counter_arg_model.add(keras.layers.Dropout(0.20))
    counter_arg_model.add(keras.layers.Flatten())
    counter_arg_model.add(keras.layers.Dense(72, activation='relu'))
    counter_arg_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # TODO: Add model compilation.
    # counter_arg_model.compile(loss=keras.losses.categorical_crossentropy,              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    return counter_arg_model
