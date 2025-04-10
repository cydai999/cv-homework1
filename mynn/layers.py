import numpy as np
from abc import abstractmethod


class Layer:
    def __init__(self):
        self.optimizable = True

    @abstractmethod
    def forward(self, X: np.ndarray):
        pass

    @abstractmethod
    def backward(self, grads: np.ndarray):
        pass

    def frozen(self):
        self.optimizable = False



class Linear(Layer):
    def __init__(self, in_dim, out_dim, weight_initialize_method=np.random.normal, weight_decay=True, weight_decay_param=1e-4):
        super().__init__()
        self.params = {
            'W':weight_initialize_method(size=(in_dim, out_dim)) * 0.1,
            'b':np.zeros((1, out_dim))
        }
        self.grads = {'W': None, 'b': None}

        self.input = None    # for backward process

        self.weight_decay = weight_decay
        self.weight_decay_param = weight_decay_param

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        """
        conduct forward process
        :param X: [batch_size, in_dim]
        :return: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.params['W'] + self.params['b']

    def backward(self, grads: np.ndarray):
        """
        calculate the grads of the parameters
        :param grads: [batch_size, out_dim]
        :return: [batch_size, in_dim]
        """
        self.grads['W'] = self.input.T @ grads
        self.grads['b'] = np.sum(grads, axis = 0, keepdims=True)
        return grads @ self.params['W'].T

    def deactivate_weight_decay(self):
        self.weight_decay = False

class Conv2D(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, input: np.ndarray):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        """
        perform ReLU function
        """
        self.input = X
        return np.where(X < 0, 0, X)

    def backward(self, grads: np.ndarray):
        """
        conduct backward process
        """
        return np.where(self.input < 0, 0, grads)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        return 1 / (1 + np.exp(-X))

    def backward(self, grads: np.ndarray):
        X = self.input
        return 1 / (2 + np.exp(X) + np.exp(-X)) * grads

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.frozen()
        self.input = None
        self.alpha = alpha

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        return np.where(X < 0, self.alpha * X, X)

    def backward(self, grads: np.ndarray):
        np.where(self.input < 0, self.alpha * grads, grads)







