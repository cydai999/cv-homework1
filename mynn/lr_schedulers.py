from abc import abstractmethod

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass

class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size=3, gamma=0.5):
        super().__init__(optimizer)
        self.step_size = step_size    # how many epoch to adjust learning rate
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count == self.step_size:
            self.optimizer.lr *= self.gamma
            self.step_count = 0

class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, step_list, gamma=0.5):
        super().__init__(optimizer)
        self.step_list = step_list
        self.gamma = gamma

    def step(self):
        self.step_count += 1
        if self.step_count in self.step_list:
            self.optimizer.lr *= self.gamma

class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma
