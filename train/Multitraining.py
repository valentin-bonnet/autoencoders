import tensorflow as tf
import numpy as np
import os
import sys

curr_dir = os.getcwd()
path = os.path.abspath(os.path.join(curr_dir, '..'))
sys.path.append(os.path.join(path, 'models'))
sys.path.append(os.path.join(path, 'utils'))
sys.path.append(os.path.join(path, 'datas'))

import image_saver
import training
import datasetLoader
import construct_model

class Multitraining():
    def __init__(self, datasets, models, optimizers, lrs, lrs_fn, epochs_max, saves_epochs, path_to_directory):
        len_list = map(len, [datasets, models, optimizers, lrs, lrs_fn, epochs_max, saves_epochs])
        self.len_max = len_list[0]
        self.trainings = []
        if not all(length == len_list[0] for length in len_list):
            print("The arguments are not all from same size")

        else:
            self.datasets = datasets
            self.models = models
            self.optimizers = optimizers
            self.lrs = lrs
            self.lrs_fn = lrs_fn
            self.epochs_max = epochs_max
            self.path_to_directory = path_to_directory
            self.saves_epochs = saves_epochs
            self.t_loss = []
            self.t_acc = []
            self.v_loss = []
            self.v_acc = []
            for i in range(self.len_max):
                train = training.Training(self.datasets[i], self.models[i], self.optimizers[i], self.lrs[i],
                                          self.lrs_fn[i], self.epochs_max[i], self.path_to_directory,
                                          self.saves_epochs[i])
                self.trainings.append(train)

    def forward(self, epochs=None, steps=None):
        for train in self.trainings:
            train.forward(epochs, steps)
            self.t_loss.append(train.t_loss)
            self.t_acc.append(train.t_acc)
            self.v_loss.append(train.v_loss)
            self.v_acc.append(train.v_acc)












