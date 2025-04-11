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
    def __init__(self, in_channel, out_channel, kernel_size, weight_initialize_method=np.random.normal, weight_decay=True, weight_decay_param=1e-4):
        super().__init__()
        self.params = {
            'W': weight_initialize_method(size=(in_channel, out_channel, kernel_size[0], kernel_size[1])),
            'b': np.zeros((out_channel, 1, 1))
        }
        self.grads = {'W': None, 'b': None}
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_param = weight_decay_param


    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        """
        input: [batch_size, in_channel, H, W]
        kernel: [in_channel, out_channel, M, N]
        output: [batch_size, out_channel, H-M+1, W-N+1] (No padding)
        """
        self.input = X
        bs, _, h, w = X.shape
        _, _, m, n = self.params['W'].shape
        output = np.zeros((bs, self.out_channel, h - m + 1, w - n + 1))
        for i in range(bs):
            for p in range(self.out_channel):
                conv_result = np.zeros(output.shape[-2:])
                for d in range(self.in_channel):
                    conv_result += self.conv(self.params['W'][d, p], X[i, d])
                output[i, p] = conv_result + self.params['b'][p]
        return output

    def backward(self, grads: np.ndarray):
        grad_w = np.zeros_like(self.params['W'])
        for d in range(grad_w.shape[0]):
            for p in range(grad_w.shape[1]):
                for i in range(self.input.shape[0]):
                    grad_w[d, p] += self.conv(grads[i, p] ,self.input[i, d])
        self.grads['W'] = grad_w

        self.grads['b'] = grads.sum(axis=(0, 2, 3))[:, None, None]

        result = np.zeros_like(self.input)
        bs = result.shape[0]
        m, n = self.params['W'].shape[-2]
        for i in range(bs):
            for d in range(self.in_channel):
                conv_result = np.zeros(result.shape[-2:])
                for p in range(self.out_channel):
                    conv_result += self.conv(self.rot(self.params['W'][d, p]), self.padding(grads[i, p], ver=m-1, hor=n-1))
                result[i ,d] = conv_result

        return result

    @staticmethod
    def conv(W, X):
        m, n = W.shape
        h, w = X.shape
        Z = np.zeros((h - m + 1, w - n + 1))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = np.sum(W * X[i: i + m, j: j+ n])
        return Z

    @staticmethod
    def rot(W):
        rot_W = np.zeros_like(W)
        m, n = W.shape
        for i in range(m):
            for j in range(n):
                rot_W[i, j] = W[m - i - 1, n - j - 1]
        return rot_W

    @staticmethod
    def padding(X, ver=0, hor=0):
        h, w = X.shape
        pad_X = np.zeros((h + 2 * ver, w + 2 * hor))
        pad_X[ver: h + ver, hor: w + hor] = X
        return pad_X

class MaxPooling(Layer):
    def __init__(self):
        super().__init__()
        self.frozen()
        self.input = None

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, X: np.ndarray):
        self.input = X
        h, w = X.shape
        assert not h % 2, 'X.shape[0] can\'t be divided by 2'
        assert not w % 2, 'X.shape[1] can\'t be divided by 2'
        output = np.zeros((h / 2, w / 2))
        for i in range(h / 2):
            for j in range(w / 2):
                output[i, j] = np.max(X[2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)])
        return output

    def backward(self, grads: np.ndarray):
        h, w = self.input.shape
        m, n = grads.shape
        result = np.zeros_like(self.input)
        assert h / 2 == m and w / 2 == n, 'Shape of grads don\'t satisfy the input'
        for i in range(m):
            for j in range(n):
                input_part = self.input[2 * i: 2 * (i + 1), 2 * j: 2 * (j + 1)]
                max_idx = np.argmax(input_part)
                result[2 * i + max_idx // 2, 2 * j + max_idx % 2] = grads[i, j]
        return result

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
        return np.where(self.input < 0, self.alpha * grads, grads)







