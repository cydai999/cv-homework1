import numpy as np

def accuracy(labels, predicts):
    assert len(labels) == len(predicts), 'size of predicts doesn\'t match labels'
    acc = np.sum(labels == predicts) / len(labels)
    return acc
