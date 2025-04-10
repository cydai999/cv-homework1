import numpy as np

def accuracy(labels, predicts):
    assert len(labels) == len(predicts), 'size of predicts doesn\'t match labels'
    acc = np.sum(labels == np.argmax(predicts, axis=1)) / len(labels)
    return acc
