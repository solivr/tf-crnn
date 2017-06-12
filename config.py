#!/usr/bin/env python
__author__ = 'solivr'


class Conf:
    def __init__(self, n_classes=None, train_batch_size=100, learning_rate=0.001, decay_rate=0.96, decay_steps='1000',
                 optimizer='rms',
                 eval_batch_size=200, max_iteration=10000, max_epochs=50, eval_interval=100,
                 save_interval=1000, file_writer=None, model_dir=None, data_set='../data', max_len=28,
                 input_shape=[32, 100], list_n_hidden=[256, 256], summary_dir='./graph'):
        self.nClasses = n_classes
        self.trainBatchSize = train_batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.optimizer= optimizer
        self.evalBatchSize = eval_batch_size
        self.maxIteration = max_iteration
        self.maxEpochs = max_epochs
        self.evalInterval = eval_interval
        self.saveInterval = save_interval
        self.fileWriter = file_writer
        self.modelDir = model_dir
        self.dataSet = data_set
        self.maxLength = max_len
        self.inputShape = input_shape
        self.imgH = self.inputShape[0]
        self.imgW = self.inputShape[1]
        try:
            self.imgC = self.inputShape[2]
        except IndexError:
            self.imgC = 1
        self.listNHidden = list_n_hidden
        self.summaryDir = summary_dir