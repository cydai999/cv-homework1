import numpy as np
from abc import abstractmethod


class NeuralNetworkModel:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
