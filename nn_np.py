"""
This file is for two-class vehicle classification
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_vehicle_data
import time
import pdb


class LogisticClassifier(object):
    def __init__(self, w_shape):
        """__init__

        :param w_shape: create w with shape w_shape using normal distribution
        """

        mean = 0
        std = 1
        self.w = np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape)

    def feed_forward(self, x):
        """feed_forward
        This function compute the output of your logistic classification model

        :param x: input

        :return result: feed forward result (after sigmoid) 
        """
        # [TODO 1.5]
        # Compute feedforward result

        result = 1/(1+np.exp(-np.dot(x, self.w)))
        return result

    def compute_loss(self, y, y_hat):
        """compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the propabilitis that the given samples belong to class 1

        :return loss: a single value
        """
        # [TODO 1.6]
        # Compute loss value (a single number)

        #loss = -np.sum((y[i]*np.log(y_hat[i])+(1-y[i])*np.log(1-y_hat[i])) for i in range(y.shape[0]))/y.shape[0]
        loss = -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
        # print(loss.shape)
        return loss

    def get_grad(self, x, y, y_hat):
        """get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y

        :return w_grad: has the same shape as self.w
        """
        # [TODO 1.7]
        # Compute the gradient matrix of w, it has the same size of w

        w_grad = np.matmul(x.T, y_hat-y)/y.shape[0]
        return w_grad

    def update_weight(self, grad, learning_rate):
        """update_weight
        Update w using the computed gradient

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        # [TODO 1.8]
        # Update w using SGD

        self.w = self.w - learning_rate*grad

    def update_weight_momentum(self, grad, learning_rate, momentum, momentum_rate):
        """update_weight with momentum
        Update w using the algorithm with momnetum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum: the array storing momentum for training w, should have the same shape as w
        :param momentum_rate: float, how much momentum to reuse after each loop (denoted as gamma in the document)
        """
        # [TODO 1.9]
        # Update w using SGD with momentum
        momentum = momentum_rate*momentum + learning_rate*grad

        self.w = self.w - momentum


def plot_loss(all_loss):
    plt.figure(1)
    plt.clf()
    plt.plot(all_loss)


def normalize_per_pixel(train_x, test_x):
    """normalize_per_pixel
    This function computes train mean and standard deviation on each pixel then applying data scaling on train_x and test_x using these computed values

    :param train_x: train images, shape=(num_train, image_height, image_width)
    :param test_x: test images, shape=(num_test, image_height, image_width)
    """
    # [TODO 1.1]
    # train_mean and train_std should have the shape of (1, image_height, image_width)
    # train_x = ...
    # test_x = ...
    mean_per_pix = np.sum(train_x, axis=0) / train_x.shape[0]
    # print(mean_per_pix)
    std_per_pix = np.sqrt(
        np.sum((x - mean_per_pix)**2 for x in train_x)/train_x.shape[0])
    # print(std_per_pix)
    for i in range(train_x.shape[0]):
        train_x[i] = np.divide(np.subtract(
            train_x[i], mean_per_pix), std_per_pix)
    for i in range(test_x.shape[0]):
        test_x[i] = np.divide(np.subtract(
            test_x[i], mean_per_pix), std_per_pix)

    return train_x, test_x


def normalize_all_pixel(train_x, test_x):
    """normalize_all_pixel
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x and test_x using these computed values

    :param train_x: train images, shape=(num_train, image_height, image_width)
    :param test_x: test images, shape=(num_test, image_height, image_width)
    """
    # [TODO 1.2]
    # train_mean and train_std should have the shape of (1, image_height, image_width)
    mean_all_pix = np.mean(train_x)
    std_all_pix = np.std(train_x)
    # print(mean_all_pix)
    for i in range(train_x.shape[0]):
        train_x[i] = (train_x[i]-mean_all_pix)/std_all_pix
    for i in range(test_x.shape[0]):
        test_x[i] = (test_x[i]-mean_all_pix)/std_all_pix

    return train_x, test_x


def reshape2D(tensor):
    """reshape_2D
    Reshape our 3D tensors to 2D. A 3D tensor of shape (num_samples, image_height, image_width) must be reshaped into (num_samples, image_height*image_width)
    """
    # [TODO 1.3]
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2])

    return tensor


def add_one(x):
    """add_one

    This function add ones as an additional feature for x
    :param x: input data
    """
    # [TODO 1.4]
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

    return x


def test(y_hat, test_y):
    """test
    Compute precision, recall and F1-score based on predicted test values

    :param y_hat: predicted values, output of classifier.feed_forward
    :param test_y: test labels
    """

    # [TODO 1.10]
    # Compute test scores using test_y and y_hat
    #precision = TP/(TP+FP)
    precision = np.sum(np.logical_and(
        y_hat > 0.5, test_y == 1))/np.sum(y_hat > 0.5)
    #recall = TP/P
    recall = np.sum(np.logical_and(y_hat > 0.5, test_y == 1)) / \
        np.sum(test_y == 1)
    f1 = 2*precision*recall/(precision+recall)
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1-score: %.3f" % f1)

    return precision, recall, f1


def generate_unit_testcase(train_x, train_y):
    train_x = train_x[0:5, :, :]
    train_y = train_y[0:5, :]

    testcase = {}
    testcase['output'] = []

    train_x_norm1, _ = normalize_per_pixel(train_x, train_x)
    train_x_norm2, _ = normalize_all_pixel(train_x, train_x)
    train_x = train_x_norm2

    testcase['train_x_norm1'] = train_x_norm1
    testcase['train_x_norm2'] = train_x_norm2

    train_x = reshape2D(train_x)
    testcase['train_x2D'] = train_x

    train_x = add_one(train_x)
    testcase['train_x1'] = train_x

    learning_rate = 0.001
    momentum_rate = 0.9

    for i in range(10):
        test_dict = {}
        classifier = LogisticClassifier((train_x.shape[1], 1))
        test_dict['w'] = classifier.w

        y_hat = classifier.feed_forward(train_x)
        loss = classifier.compute_loss(train_y, y_hat)
        grad = classifier.get_grad(train_x, train_y, y_hat)
        classifier.update_weight(grad, 0.001)
        test_dict['w_1'] = classifier.w

        momentum = np.ones_like(grad)
        classifier.update_weight_momentum(
            grad, learning_rate, momentum, momentum_rate)
        test_dict['w_2'] = classifier.w

        test_dict['y_hat'] = y_hat
        test_dict['loss'] = loss
        test_dict['grad'] = grad

        testcase['output'].append(test_dict)

    np.save('./data/unittest', testcase)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


if __name__ == "__main__":
    np.random.seed(2018)

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())

    # Normalize our data: choose one of the two methods before training
    #train_x, test_x = normalize_all_pixel(train_x, test_x)
    train_x, test_x = normalize_per_pixel(train_x, test_x)

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)

    num_i = train_x.shape[1]  # number of inputs
    num_h = 8  # number of neurons of hidden layer
    num_o = 1  # number of outputs

    lr = 0.01  # 0.1 , 1 -> learning rate

    W1 = np.random.normal(0, 1, size=(num_h, num_i))  # 128x4096
    b1 = np.random.normal(0, 1, size=(num_h, 1))      # 128x1
    W2 = np.random.normal(0, 1, size=(num_o, num_h))  # 1x128
    b2 = np.random.normal(0, 1, size=(num_o, 1))      # 1x1

    for i in range(1000):
        # feedforward
        Z1 = W1.dot(train_x.T) + b1  # 2x4
        A1 = sigmoid(Z1)                        # 1/(1+np.exp(-Z1))  # 2x4

        Z2 = W2.dot(A1) + b2  # 1x4
        A2 = sigmoid(Z2)  # 1/(1+np.exp(-Z2))  # 1x4 A(L)
        # print(A2.shape)
        # print(train_y.shape)
        # print(Z1)
        # print(A1)
        # print(Z2)
        # print(A2)
        # # back propagation
        E2 = A2 - train_y.T  # 1x4 
        # print(E2)
        derivative_A2 = sigmoid_derivative(A2)   # np.multiply(A2, (1-A2))
        D2 = np.multiply(E2, derivative_A2)  # 1x4 element-wise product #dz2
        # print(D2)
        E1 = W2.T.dot(D2)#dw2
        # print(E1)
        derivative_A1 = sigmoid_derivative(A1)
        D1 = np.multiply(E1, derivative_A1)#dz1
        # print(D1)

        # update weights
        W2 = W2 - lr*D2.dot(A1.T)
        b2 = b2 - lr*np.sum(D2, axis=1, keepdims=True)
        W1 = W1 - lr*D1.dot(train_x)
        b1 = b1 - lr*np.sum(D1, axis=1, keepdims=True)

        cost = -np.sum(train_y.T*np.log(A2+0.00001) +
                       (1-train_y.T)*np.log(1-A2+0.00001))
        cost = cost/num_train
        if i % 10 == 0:
            print(cost)

    # inference
    Z1 = W1.dot(test_x.T) + b1  # 2x4
    A1 = sigmoid(Z1)                        # 1/(1+np.exp(-Z1))  # 2x4

    Z2 = W2.dot(A1) + b2  # 1x4
    # 1/(1+np.exp(-Z2))  # 1x4 A(L)        Z1 = W1.dot(train_x.T) + b1  # 2x4
    A2 = sigmoid(Z2)

    test(A2, test_y.T)
