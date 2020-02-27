import sys
import os
import tensorflow as tf
curr_dir = os.getcwd()
sys.path.append(curr_dir)


import cifarLoader
import DAVISLoader
import ImagenetResizedLoader

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset():
    def __init__(self, dataset_name, batch_size=128, buffer_size=10000, use_Lab=False):
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.shape = None
        self.size = None
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.name = dataset_name
        if dataset_name == 'DAVIS':
            self.train_ds = DAVISLoader.get_dataset(
                '/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                'train')
            self.val_ds = DAVISLoader.get_dataset(
                '/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                'val')
            shape = 32
        elif dataset_name == 'cifar10':
            if use_Lab:
                self.train_ds, self.val_ds = cifarLoader.cifarloaderLab()
            else:
                self.train_ds, self.val_ds = cifarLoader.cifarloader()
            self.shape = 32

        elif dataset_name == 'imagenetresized64':
            self.train_ds, self.val_ds = ImagenetResizedLoader.imagenetresized64loaderLab()
            self.shape = 64

        else:
            print("No good dataset selected")

        if self.train_ds is not None:
            self.train_ds = self.train_ds.shuffle(buffer_size=self.buffer_size).batch(self.batch_size).prefetch(AUTOTUNE)
        if self.val_ds is not None:
            self.val_ds = self.val_ds.shuffle(buffer_size=self.buffer_size).batch(self.batch_size).prefetch(AUTOTUNE)
        if self.test_ds is not None:
            self.test_ds = self.test_ds.shuffle(buffer_size=self.buffer_size).batch(self.batch_size).prefetch(AUTOTUNE)

