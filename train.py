import json
import os
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
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

train_data_list = []
train_labels_list = []

for i in range(5):
    path = f'data_batch_{i+1}'
    dict = unpickle(os.path.join(dataset_dir, path))
    train_data_list.append(dict[b'data'])
    train_labels_list.append(dict[b'labels'])

train_data = np.concatenate(train_data_list, axis=0)
train_labels = np.concatenate(train_labels_list, axis=0)

idx = np.random.permutation(train_data.shape[0])
train_data, train_labels = train_data[idx], train_labels[idx]
valid_data, valid_labels = train_data[:10000], train_labels[:10000]
train_data, train_labels = train_data[10000:], train_labels[10000:]

# valid_data, valid_labels = train_data[:1000], train_labels[:1000]
# train_data, train_labels = train_data[10000: 20000], train_labels[10000: 20000]

# normalize
train_data = train_data / train_data.max()
valid_data = valid_data / valid_data.max()

valid_set = [valid_data, valid_labels]
train_set = [train_data, train_labels]

test_path = 'test_batch'
test_dict = unpickle(os.path.join(dataset_dir, test_path))
test_data = test_dict[b'data']
test_data = test_data / test_data.max()
test_labels = test_dict[b'labels']
test_set = [test_data, test_labels]

# get parameter from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', '-hs', type=int, default=1000, help='the number of neurons of the hidden layer')
parser.add_argument('--act_func', '-a', type=str, default='LeakyReLU', help='activation function')
parser.add_argument('--weight_decay_param', '-wd', type=float, default=1e-5, help='parameter of weight decay')
parser.add_argument('--init_lr', '-lr', type=float, default=1e-2, help='initialized learning rate')
parser.add_argument('--step_size', '-s', type=int, default=5, help='learning rate decay period')
parser.add_argument('--gamma', '-g', type=float, default=0.1, help='parameter of learning rate decay')
parser.add_argument('--batch_size','-bs', type=int, default=32)
parser.add_argument('--epoch', '-e', type=int, default=10)
parser.add_argument('--log_iter', '-l', type=int, default=100, help='period of print loss and accuracy')

args = parser.parse_args()

# init model
hidden_size = args.hidden_size
act_func = args.act_func
weight_decay_param = args.weight_decay_param
init_lr = args.init_lr
step_size = args.step_size
gamma = args.gamma
batch_size = args.batch_size

size_list = [3072, hidden_size, 10]
weight_decay_list = [weight_decay_param, weight_decay_param]

model = nn.models.MLPModel(size_list=size_list, act_func=act_func, weight_decay_list=weight_decay_list)
# optimizer = nn.optimizers.SGD(model, init_lr=init_lr)
optimizer = nn.optimizers.SGDMomentum(model, init_lr=init_lr)
lr_scheduler = nn.lr_schedulers.StepLR(optimizer, step_size=step_size, gamma=gamma)
metric = nn.metric.accuracy
loss_fn = nn.loss_fn.CrossEntropyLoss(model)
runner = nn.runner.Runner(model, loss_fn, metric, batch_size=batch_size, optimizer=optimizer, lr_scheduler=lr_scheduler)

# train
epoch = args.epoch
# save_dir = f'./saved_models/{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}'
save_dir = f'./saved_models/test_train'
log_iter = args.log_iter
print('[Train]Begin training...')
runner.train(train_set, valid_set, epoch, save_dir, log_iter)
print('[Train]Training completed!')

# test
test_data, test_labels = test_set
print('[Test]Begin testing...')
loss, score = runner.eval(test_data, test_labels)
print('[Test]Testing completed!')
print(f'[Test]loss:{loss}, score:{score}')

# save result
result_path = os.path.join(save_dir, 'testing_result.json')
result = {'loss': loss, 'score': score}
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f)

# plot
fg, axis = plt.subplots(1, 2)
axis[0].plot(runner.train_loss, label='train_loss', color='r')
axis[0].plot(runner.valid_loss, label='valid_loss', color='b')
axis[0].set_xlabel('iteration')
axis[0].set_ylabel('loss')
axis[0].legend(loc='upper right')

axis[1].plot(runner.train_score, label='train_score', color='r')
axis[1].plot(runner.valid_score, label='valid_score', color='b')
axis[1].set_xlabel('iteration')
axis[1].set_ylabel('score')
axis[1].legend(loc='upper right')

fg.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_accuracy_plot.png'))

# save_params
params = {'model':{
              'size_list': size_list,
              'act_func': act_func,
              'weight_decay_list': weight_decay_list
          },
          'optimizer':{
              'type': 'momentum',
              'init_lr': init_lr
          },
          'lr_scheduler':{
              'type': 'stepLR',
              'step_size': step_size,
              'gamma': gamma
          },
          'metric':{
              'type': 'accuracy'
          },
          'loss function':{
              'type': 'cross entropy'
          },
          'train':{
              'batch_size': batch_size,
              'epoch': epoch,
              'log_iter': log_iter
          }
}

json_path = os.path.join(save_dir, 'params.json')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(params, f)


