import numpy as np
from abc import abstractmethod

class LossFunc:
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

class CrossEntropyLoss(LossFunc):
    def __init__(self, model):
        super().__init__(model)

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