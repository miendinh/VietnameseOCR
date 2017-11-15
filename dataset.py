#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import sys
from config import *

class DataSet:
    def __init__(self, test_prob = 0.2, one_hot = True):
        self.test_prob = test_prob
        self.one_hot = one_hot
        self.X_test = []
        self.Y_test = []
        self.curr_training_step = 0
        self.curr_test_step = 0
        self.line_offset = self.get_line_offset()
        self.train_idx, self.test_idx = self.shuffle_data_set()  
   

    def to_one_hot(self, X):
        one_hot = np.zeros((len(X), NO_LABEL))
        for i in range(len(X)):
            np.put(one_hot[i, :], X[i], 1)        

        return one_hot

    def shuffle_data_set(self):
        idx = list(range(SUM_SAMPLES))

        for i in range(5):
            random.shuffle(idx)

        test_size = int(self.test_prob * SUM_SAMPLES)

        test_idx = idx[0:test_size]
        train_idx = idx[test_size:SUM_SAMPLES]
        print('Done shuffle dataset!');
        return (train_idx, test_idx)

    def next_batch(self, batch_size):
        idx = self.train_idx[self.curr_training_step*batch_size:self.curr_training_step*batch_size + batch_size]
        data = []

        with open(DATASET_FILE, 'r') as ds:
            for i in range(len(idx)):
                ds.seek(self.line_offset[idx[i]])
                line = ds.readline().strip()
                data.append(line.split(','))

        X_train_bs, Y_train_bs = self.split_image_label(data)

        self.curr_training_step = self.curr_training_step + 1
        self.curr_training_step = self.curr_training_step if (self.curr_training_step*batch_size < len(self.train_idx)) else 0

        return (X_train_bs, Y_train_bs)

    def next_batch_test(self, batch_size):
        idx = self.test_idx[self.curr_test_step*batch_size:self.curr_test_step*batch_size + batch_size]
        data = []
        debug_offset = None
        debug_idx = None
        with open(DATASET_FILE, 'r') as ds:
            for i in range(len(idx)):
                debug_idx = idx[i]
                debug_offset = self.line_offset[idx[i]]
                ds.seek(self.line_offset[idx[i]])
                line = ds.readline().strip()
                data.append(line.split(','))

        X_test_bs, Y_test_bs = self.split_image_label(data)

        self.curr_test_step = self.curr_test_step + 1
        self.curr_test_step = self.curr_test_step if (self.curr_test_step*batch_size < len(self.test_idx)) else 0

        return (X_test_bs, Y_test_bs)

    def split_image_label(self, data):
        if(type(data) is list):
            data = np.asarray(data, dtype=np.float32)

        image = data[:, :-1]
        label = data[:, -1]
        if self.one_hot: 
            return (image, self.to_one_hot(label))
        else:
            return (image, label)

    def get_test_set(self):

        if len(self.X_test) > 0 and len(self.Y_test) > 0:
            return (self.X_test, Y_test)

        idx = self.test_idx
        data = []
        with open(DATASET_FILE, 'r') as ds:
            for i in range(len(idx)):
                ds.seek(self.line_offset[idx[i]])
                line = ds.readline().replace('\n', '')
                data.append(line.split(','))


        self.X_test, self.Y_test = self.split_image_label(data)

        return (self.X_test, self.Y_test)

    def get_line_offset(self):
        line_offset = []
        offset = 0
        with open(DATASET_FILE, 'r') as ds:
            for line in ds:
                line_offset.append(offset)
                offset += len(line)

        return line_offset


#if __name__ == "__main__":
#   prepare = DataSet()