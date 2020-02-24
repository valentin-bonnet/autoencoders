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
import datasetLoader
import construct_model

class Training():
    def __init__(self, dataset, model, optimizer, lr, lr_fn, epoch_max, path_to_directory, save_epochs):
        self.train_ds = dataset.train_ds
        self.train_size = dataset.train_size
        self.val_ds = dataset.val_ds
        self.val_size = dataset.val_size
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.lr_fn = lr_fn
        self.epoch_max = epoch_max
        self.name = '_'.join(filter(None, [dataset.name, model.description]))
        self.path = os.path.join(path_to_directory, self.name)
        self.ckpt_path = os.path.join(self.path, 'ckpts')
        self.train_path = os.path.join(self.path, 'training')
        self.val_path = os.path.join(self.path, 'validation')
        self.img_path = os.path.join(self.path, 'imgs')
        self.save_epochs = save_epochs

        self.t_loss = None
        self.t_acc = None
        self.v_loss = None
        self.v_acc = None

        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        if not os.path.isdir(self.train_path):
            os.makedirs(self.train_path)

        if not os.path.isdir(self.val_path):
            os.makedirs(self.val_path)

        self.current_epoch = tf.Variable(0)
        self.ckpt = tf.train.Checkpoint(step=self.current_epoch, optimizer=self.optimizer, net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=self.epoch_max // self.save_epochs)

        self.load()


    def forward(self, epochs=None, steps=None):
        if epochs is None or epochs + int(self.ckpt.step) > self.epoch_max:
            epochs = self.epoch_max
        if steps is None:
            steps = 0
        if steps >= self.train_size:
            steps = self.train_size-1

        t_loss_mean = tf.keras.metrics.Mean(name='t_loss')
        t_acc_mean = tf.keras.metrics.Mean(name='t_acc')
        v_loss_mean = tf.keras.metrics.Mean(name='v_loss')
        v_acc_mean = tf.keras.metrics.Mean(name='v_acc')

        starting_epoch = self.ckpt.step
        for epoch in range(starting_epoch, epochs+1):
            len_train = self.train_size
            progbar = tf.keras.utils.Progbar(len_train)

            self.lr = self.lr_fn(self.lr, epoch)
            self.optimizer.lr = self.lr

            # One epoch on TRAIN dataset
            for i, train_x in enumerate(self.train_ds):
                progbar.update(i + 1)
                t_loss_mean(self.model.compute_apply_gradients(train_x, self.optimizer))
                t_acc_mean(self.model.compute_accuracy(train_x))

            # One epoch on VALIDATION dataset
            for i, val_x in enumerate(self.val_ds):
                v_loss_mean(self.model.compute_loss(val_x))
                v_acc_mean(self.model.compute_accuracy(val_x))

            self.t_loss.append(t_loss_mean.result().numpy())
            self.t_acc.append(t_acc_mean.result().numpy())
            self.v_loss.append(v_loss_mean.result().numpy())
            self.v_acc.append(v_acc_mean.result().numpy())

            # Create temp image of loss
            image_saver.curves([self.t_loss, self.v_loss], ['Training', 'Validation'], 'epochs', 'loss', 'training_validation_loss', self.img_path)
            image_saver.curves([self.t_acc, self.v_acc], ['Training', 'Validation'], 'epochs', 'accuracy', 'training_validation_accuracy', self.img_path)



            img_name = 'epoch_'+str(epoch)
            for val_x in self.val_ds.take(1):

                image_saver.compare_images(val_x, self.model.reconstruct(val_x), img_name, self.img_path)

            t_loss_mean.reset_states()
            t_acc_mean.reset_states()
            v_loss_mean.reset_states()
            v_acc_mean.reset_states()

            if epoch % self.save_epochs or epoch == epochs:
                self.save()
            self.ckpt.step.assign_add(1)


    def save(self):
        #Save info about loss and acuracy of training and validation dataset
        t_loss_path = os.path.join(self.train_path, 'loss.npy')
        t_acc_path = os.path.join(self.train_path, 'accuracy.npy')
        v_loss_path = os.path.join(self.val_path, 'loss.npy')
        v_acc_path = os.path.join(self.val_path, 'accuracy.npy')

        #Save model with checkpoint
        save_path = self.ckpt_manager.save()
        np.save(t_loss_path, np.asarray(self.t_loss))
        np.save(t_acc_path, np.asarray(self.t_acc))
        np.save(v_loss_path, np.asarray(self.v_loss))
        np.save(v_acc_path, np.asarray(self.v_acc))
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))



    def load(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        t_loss_path = os.path.join(self.train_path, 'loss.npy')
        t_acc_path = os.path.join(self.train_path, 'accuracy.npy')
        v_loss_path = os.path.join(self.val_path, 'loss.npy')
        v_acc_path = os.path.join(self.val_path, 'accuracy.npy')

        self.t_loss = np.load(t_loss_path).tolist() if os.path.isfile(t_loss_path) else []
        self.t_acc = np.load(t_acc_path).tolist() if os.path.isfile(t_acc_path) else []
        self.v_loss = np.load(v_loss_path).tolist() if os.path.isfile(v_loss_path) else []
        self.v_acc = np.load(v_acc_path).tolist() if os.path.isfile(v_acc_path) else []












