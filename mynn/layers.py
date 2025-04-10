import numpy as np
from abc import abstractmethod


class Layer:
    def __init__(self):
        self.optimizable = True

    @abstractmethod
    def forward(self, input: np.ndarray):
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

class Logistic(Layer):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, input: np.ndarray):
        pass

    def forward(self, input: np.ndarray):
        pass

    def backward(self, grads: np.ndarray):
        pass

class CrossEntropyLoss(Layer):
    def __init__(self, model):
        super().__init__()
        self.optimizable = False

        self.model = model

        self.input = None
        self.labels = None
        self.grads = None

    def __call__(self, labels, predicts):
        return self.forward(labels, predicts)

    def forward(self, labels, predicts):
        """
        Calculate loss with a softmax layer
        :param predicts: [batch_size, d]
        :param labels: [batch_size, ]
        :return: average loss of the batch
        """
        bs = predicts.shape[0]
        self.input = predicts
        self.labels = labels
        # softmax
        P = self.softmax(predicts)
        # calculate loss
        loss = - np.sum(np.log(P[np.arange(bs), labels] + 1e-8))
        return loss / bs

    def backward(self):
        """
        calculate the grads from loss to the input, then pass it to the model to start backpropagation
        """
        bs = self.input.shape[0]
        dim = self.input.shape[1]
        # calculate the loss
        P = self.softmax(self.input)
        onehot_matrix = np.zeros((bs, dim))
        onehot_matrix[np.arange(bs), self.labels] = 1
        self.grads = (P - onehot_matrix) / bs
        # pass it to the model
        self.model.backward(self.grads)

    @staticmethod
    def softmax(X):
        X_max = np.max(X, axis=1, keepdims=True)    # avoid overflow
        X_exp = np.exp(X - X_max)
        return X_exp / np.sum(X_exp, axis=1, keepdims=True)





