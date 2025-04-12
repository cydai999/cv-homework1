import os
import pickle
import numpy as np
import argparse

import mynn as nn

# set seed
np.random.seed(123)

# load data
dataset_dir = './dataset/cifar-10-python/cifar-10-batches-py'

def unpickle(path):
    with open(path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

test_path = 'test_batch'
test_dict = unpickle(os.path.join(dataset_dir, test_path))
test_data = test_dict[b'data']
test_data = test_data / test_data.max()
test_labels = test_dict[b'labels']
test_set = [test_data, test_labels]

# load model
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', '-p', type=str, default='./saved_models/best_model/models/best_model.pickle')
args = parser.parse_args()

model_path = args.model_path

model = nn.models.MLPModel(save_dir=model_path)

# test
loss_fn = nn.loss_fn.CrossEntropyLoss(model)
metric = nn.metric.accuracy
runner = nn.runner.Runner(model, loss_fn=loss_fn, metric=metric)

test_data, test_labels = test_set
print('[Test]Begin testing...')
loss, score = runner.eval(test_data, test_labels)
print('[Test]Testing completed!')
print(f'[Test]loss:{loss}, score:{score}')