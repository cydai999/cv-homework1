import os.path

import numpy as np
from tqdm import tqdm

class Runner:
    def __init__(self, model, optimizer, loss_fn, metric, batch_size, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metric = metric
        self.batch_size = batch_size
        self.best_score = 0

        self.train_loss = []
        self.train_score = []

        self.valid_loss = []
        self.valid_score = []

    def train(self, train_set, valid_set, epoch=5, save_dir=None, log_iter=100):
        if not save_dir:
            save_dir = 'best_model'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_model_save_path = None

        for k in range(epoch):
            # permute the data
            train_X, train_y = train_set
            idx = np.random.permutation(train_X.shape[0])
            train_X = train_X[idx]
            train_y = train_y[idx]

            # begin training
            for i in range(train_X.shape[0] // self.batch_size + 1):
                X = train_X[self.batch_size * i: self.batch_size * (i + 1)]
                y = train_y[self.batch_size * i: self.batch_size * (i + 1)]

                train_loss, train_score = self.eval(X, y)
                self.train_loss.append(train_loss)
                self.train_score.append(train_score)

                self.loss_fn.backward()

                self.optimizer.step()

                valid_loss, valid_score = self.eval(valid_set[0], valid_set[1])
                self.valid_loss.append(valid_loss)
                self.valid_score.append(valid_score)

                if i % log_iter == 0:
                    print(f'epoch:{k}, iter:{i}'
                          f'train_loss:{train_loss}, train_score:{train_score}'
                          f'valid_loss:{valid_loss}, valid_score:{valid_score}')

            if self.lr_scheduler:
                self.lr_scheduler.step()

            save_path = os.path.join(save_dir, f'epoch_{k}')
            self.model.save_model(save_path)

            if self.valid_score[-1] > self.best_score:
                print(f'Best score has been updated:{self.best_score:.5f}->{self.valid_score[-1]:.5f}')
                best_model_save_path = os.path.join(save_dir, f'best_model(epoch{k}).pickle')
                self.best_score = self.valid_score[-1]

        if best_model_save_path:
            self.model.save_model(best_model_save_path)


    def eval(self, data, labels):
        predicts = self.model(data)
        loss = self.loss_fn(predicts)
        score = self.metric(labels, predicts)
        return loss, score