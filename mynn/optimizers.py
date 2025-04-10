from abc import abstractmethod

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

class Momentum(Optimizer):
    def __init__(self, model, init_lr):
        super().__init__(model, init_lr)

    def step(self):
        pass
