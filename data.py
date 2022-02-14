import os
import cv2
import tensorflow as tf

def preproc(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = grayscale/255.0
    return grayscale

def get_data(dir_path, label):

    X = []
    y = []

    for filename in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, filename))
        img = preproc(img)
        X.append(img)
        y.append(label)

    return X, y


def get_train():

    train_X_0, train_y_0 = get_data(dir_path='dataset/toy_train/circle', label=0)
    train_X_1, train_y_1 = get_data(dir_path='dataset/toy_train/rectangle', label=1)
    
    train_X, train_y = tf.convert_to_tensor(train_X_0+train_X_1), tf.convert_to_tensor(train_y_0+train_y_1)

    print('DONE GETTING TRAIN')

    return train_X, train_y

def get_test():

    test_X_0, test_y_0 = get_data(dir_path='dataset/toy_val/circle', label=0)
    test_X_1, test_y_1 = get_data(dir_path='dataset/toy_val/rectangle', label=1)
    
    test_X, test_y = tf.convert_to_tensor(test_X_0+test_X_1), tf.convert_to_tensor(test_y_0+test_y_1)

    print('DONE GETTING VALIDATION')

    return test_X, test_y


if __name__ == '__main__':
    train_X, train_y = get_train()
    test_X, test_y = get_test()
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)