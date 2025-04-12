import pickle
from .layers import *
from abc import abstractmethod


class NeuralNetworkModel:
    def __init__(self):
        self.layer_list = []

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

class MLPModel(NeuralNetworkModel):
    def __init__(self, size_list: list[int]=None, act_func=None, **kwargs):
        super().__init__()
        self.size_list = size_list
        self.layer_list = []
        self.act_func = act_func
        self.act_func_map = {'ReLU': ReLU(), 'Sigmoid': Sigmoid(), 'LeakyReLU': LeakyReLU()}
        self.weight_decay_list = kwargs.get('weight_decay_list')
        assert not self.weight_decay_list or len(self.weight_decay_list) == len(self.size_list) - 1, 'weight decay doesn\'t match'

        self.save_dir = kwargs.get('save_dir')

        if self.save_dir:
            self.load_model(self.save_dir)
        else:
            for i in range(len(size_list) - 1):
                layer = Linear(self.size_list[i], self.size_list[i+1])
                # add weight decay
                if self.weight_decay_list and layer.weight_decay:
                    layer.weight_decay_param = self.weight_decay_list[i]
                self.layer_list.append(layer)
                # add activate layer(except the last layer)
                if i < len(size_list) - 2:
                    try:
                        self.layer_list.append(self.act_func_map[self.act_func])
                    except ValueError:
                        print('activate function not been finished')

    def __call__(self, input: np.ndarray):
        return self.forward(input)

    def forward(self, input: np.ndarray):
        assert self.size_list and self.act_func, 'Model has not been correctly initialized, try using load_model method or directly provide size_list and act_func'
        for layer in self.layer_list:
            input = layer(input)
        return input

    def backward(self, grads: np.ndarray):
        for layer in reversed(self.layer_list):
            grads = layer.backward(grads)

    def load_model(self, save_dir):
        with open(save_dir, 'rb') as f:
            param_list = pickle.load(f)

        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            layer = Linear(self.size_list[i], self.size_list[i+1])
            layer.params['W'] = param_list[i + 2]['W']
            layer.params['b'] = param_list[i + 2]['b']
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_param = param_list[i + 2]['weight_decay_param']
            self.layer_list.append(layer)
            if i < len(self.size_list) - 2:
                self.layer_list.append(self.act_func_map[self.act_func])

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layer_list:
            if layer not in self.act_func_map.values():
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'weight_decay_param': layer.weight_decay_param
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)




