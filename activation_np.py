import numpy as np


def sigmoid(x):
    """sigmoid
    TODO: 
    Sigmoid function. Output = 1 / (1 + exp(-x)).
    :param x: input
    """
    #[TODO 1.1]
    return 1/(1+np.exp(-x))


def sigmoid_grad(a):
    """sigmoid_grad
    TODO:
    Compute gradient of sigmoid with respect to input. g'(x) = g(x)*(1-g(x))
    :param a: output of the sigmoid function
    """
    #[TODO 1.1]
    output=sigmoid(a)*(1-sigmoid(a))
    return output


def reLU(x):
    """reLU
    TODO:
    Rectified linear unit function. Output = max(0,x).
    :param x: input
    """
    #[TODO 1.1]
    x[x<0]=0
    return x


def reLU_grad(a):
    """reLU_grad
    TODO:
    Compute gradient of ReLU with respect to input
    :param x: output of ReLU
    """
    #[TODO 1.1]
    a[a>=0]=1
    return a


def tanh(x):
    """tanh
    TODO:
    Tanh function.
    :param x: input
    """
    #[TODO 1.1]
    return np.tanh(x)


def tanh_grad(a):
    """tanh_grad
    TODO:
    Compute gradient for tanh w.r.t input
    :param a: output of tanh
    """
    #[TODO 1.1]
    grad=1-(tanh(a))**2
    return grad


def softmax(x):
    """softmax
    TODO:
    Softmax function.
    :param x: input
    """
    # z_max = np.amax(x, axis=1,keepdims=True)
    # zmax = np.subtract(x,z_max,dtype=np.float64)
    soft = np.exp(x)
    output = soft / np.sum(soft,axis=1,keepdims=True)
    return output


def softmax_minus_max(x):
    """softmax_minus_max
    TODO:
    Stable softmax function.
    :param x: input
    """
    z_max = np.amax(x, axis=1,keepdims=True)
    zmax = np.subtract(x,z_max,dtype=np.float64)
    soft = np.exp(zmax)
    output = soft / np.sum(soft,axis=1,keepdims=True)
    return output
