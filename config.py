#!/usr/bin/env python
__author__ = 'solivr'


class Conf:
    def __init__(self, n_class=None, train_batch_size=100, learning_rate=0.001, test_batch_size=200, max_iteration=10000,
                 display_interval=200, test_interval=100, model_file=None, data_set='../data', max_len=28,
                 input_shape=[32, 100], list_n_hidden=[256, 256], summary_dir='./graph'):
        self.nClasses = n_class
        self.trainBatchSize = train_batch_size
        self.learning_rate = learning_rate
        self.testBatchSize = test_batch_size
        self.maxIteration = max_iteration
        self.displayInterval = display_interval
        self.testInterval = test_interval
        self.modelParFile = model_file
        self.dataSet = data_set
        self.maxLength = max_len
        self.input_shape = input_shape
        self.list_n_hidden = list_n_hidden
        self.summary_dir = summary_dir


# class Config:
#     def __init__(self):
#         self.nClasses = 36
#         self.trainBatchSize = 64
#         self.evalBatchSize = 200
#         self.testBatchSize = 10
#         self.maxIteration = 2000000
#         self.displayInterval = 1
#         self.evalInterval = 10
#         self.testInterval = 20
#         self.saveInterval = 50000
#         self.modelDir = os.path.abspath(os.path.join('..', 'model', 'ckpt'))
#         # self.dataSet = os.path.join('..', 'data', 'Synth')
#         # self.auxDataSet = os.path.join('..', 'data', 'aux_Synth')
#         self.dataSet = os.path.join('..', 'data', 'IIIT5K')
#         self.maxLength = 24
#         self.trainLogPath = os.path.abspath(os.path.join('..', 'model', 'log'))
#
#         self.input_shape = [32, 100]
#         self.list_n_hidden = [256, 256]

