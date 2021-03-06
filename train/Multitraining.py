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
    def __init__(self, dataset, batch_size, models, optimizers, lrs, lrs_fn, steps_max, saves_steps, path_to_directory, step_is_epoch):
        len_list = list(map(len, [models, optimizers, lrs, lrs_fn, steps_max, saves_steps]))
        self.dataset = dataset
        self.len_max = len_list[0]
        self.trainings = []
        if not all(length == len_list[0] for length in len_list):
            print("All arguments does not have the same size")

        else:
            self.dataset = dataset
            self.batch_size = batch_size
            self.models = models
            self.optimizers = optimizers
            self.lrs = lrs
            self.lrs_fn = lrs_fn
            self.steps_max = steps_max
            self.path_to_directory = path_to_directory
            self.saves_steps = saves_steps
            self.step_is_epoch = step_is_epoch
            self.t_loss = []
            self.t_acc = []
            self.v_loss = []
            self.v_acc = []
            for i in range(self.len_max):
                train = training.Training(self.dataset, self.batch_size, self.models[i], self.optimizers[i], self.lrs[i],
                                          self.lrs_fn[i], self.steps_max[i], self.path_to_directory,
                                          self.saves_steps[i], self.step_is_epoch)
                self.trainings.append(train)

    def forward(self, epochs=None, steps=None):
        for train in self.trainings:
            print("Train")
            train.forward()
            self.t_loss.append(train.t_loss)
            self.t_acc.append(train.t_acc)
            self.v_loss.append(train.v_loss)
            self.v_acc.append(train.v_acc)

        #for val_x in self.dataset.val_ds.batch(self.batch_size, drop_remainder=True).take(1):
        #    for train in self.trainings:
        #        image_saver.compare_images(val_x[0], train.model.reconstruct(val_x)[0], 'result_'+train.name, self.path_to_directory)

        #image_saver.curves([self.t_acc[0], self.t_acc[1], self.t_acc[2], self.t_acc[3], self.t_acc[4], self.t_acc[5]], ['128', '256', '512', '1024', '2048', '4096'],
        #                   'training_validation_accuracy', self.path_to_directory, 'Epochs', 'Accuracy (L2)')












