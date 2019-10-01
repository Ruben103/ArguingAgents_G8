import tensorflow as tf


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
        if (train_labels[train_idx] == 0):
            filtered_train_images.append(train_images[train_idx])
            filtered_train_labels.append(train_labels[train_idx])
            train_ctr_0 += 1
        if (train_labels[train_idx] == 1):
            filtered_train_images.append(train_images[train_idx])
            filtered_train_labels.append(train_labels[train_idx])
            train_ctr_1 += 1
    print(f"Number of train images:\n 0: {train_ctr_0}\t 1: {train_ctr_1}")

    for test_idx in range(0, len(test_labels)):
        if (test_labels[test_idx] == 0):
            filtered_test_images.append(test_images[test_idx])
            filtered_test_labels.append(test_labels[test_idx])
            test_ctr_0 += 1
        if (test_labels[test_idx] == 1):
            filtered_test_images.append(test_images[test_idx])
            filtered_test_labels.append(test_labels[test_idx])
            test_ctr_1 += 1
    print(f"Number of test images:\n 0: {test_ctr_0}\t 1: {test_ctr_1}")
    # TODO: Add a try catch to check if data is always returned with expected shapes and sizes
    return filtered_train_images, filtered_train_labels, filtered_test_images, filtered_test_labels
