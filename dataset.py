#!/usr/bin/env python
__author__ = 'solivr'

class Dataset:
    def __init__(self, config):
        self.imgH = config.imgH
        self.imgW = config.imgW
        self.imgC = config.imgC
        self.nSamples

    def nextBatch(self, batch_size):
        #
