from abc import abstractmethod
import numpy as np

class Optimizer:
    def __init__(self, model, init_lr):
        self.model = model
        self.lr = init_lr

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, model, init_lr):
        super().__init__(model, init_lr)

    def step(self):
        for layer in self.model.layer_list:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay and key != 'b':    # do not perform weight decay on bias
                        layer.params[key] *= (1 - self.lr * layer.weight_decay_param)
                    layer.params[key] -= self.lr * layer.grads[key]

class SGDMomentum(Optimizer):
    def __init__(self, model, init_lr, momentum=0.9):
        super().__init__(model, init_lr)
        self.momentum = momentum
        self.velocities = {}

    def step(self):
        for layer in self.model.layer_list:
            if layer.optimizable:
                for key in layer.params.keys():
                    param = layer.params[key]
                    grad = layer.grads[key]
                    param_id = id(param)

                    # initialize
                    if param_id not in self.velocities:
                        self.velocities[param_id] = np.zeros_like(param)

                    velocity = self.velocities[param_id]

                    # weight decay
                    if layer.weight_decay and key != 'b':
                        param *= (1 - self.lr * layer.weight_decay_param)

                    # update momentum
                    velocity[:] = self.momentum * velocity + grad

                    # update parameters
                    param -= self.lr * velocity
