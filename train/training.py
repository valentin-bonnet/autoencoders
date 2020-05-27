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

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Training():
    def __init__(self, dataset, batch_size, model, optimizer, lr, lr_fn, epoch_max, path_to_directory, save_steps, step_is_epoch, is_seq=False):
        self.redone = False
        self.batch_size = batch_size
        if self.redone:
            self.train_ds = dataset.train_ds.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
            self.val_ds = dataset.val_ds.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        else:
            self.train_ds = dataset.train_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
            self.val_ds = dataset.val_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
            self.test_ds = dataset.test_ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        self.train_size = dataset.train_size//batch_size

        self.val_size = dataset.val_size//batch_size

        self.model = model
        #self.model_view = model_view
        self.optimizer = optimizer
        self.lr = lr
        self.lr_fn = lr_fn
        self.epoch_max = epoch_max
        self.name = '_'.join(filter(None, [dataset.name, model.description]))
        self.path = os.path.join(path_to_directory, self.name)
        self.ckpt_path = os.path.join(self.path, 'ckpts')
        self.ckpt_resnet_path = os.path.join(self.ckpt_path, 'resnet')
        self.ckpt_rkn_encoder_path = os.path.join(self.ckpt_path, 'rkn_encoder')
        self.ckpt_rkn_score_path = os.path.join(self.ckpt_path, 'rkn_score')
        self.train_path = os.path.join(self.path, 'training')
        self.val_path = os.path.join(self.path, 'validation')
        self.img_path = os.path.join(self.path, 'imgs')
        self.save_steps = save_steps
        self.step_is_epoch = step_is_epoch
        self.is_seq = is_seq

        self.t_loss = None
        self.t_acc = None
        self.v_loss = None
        self.v_acc = None

        if not self.step_is_epoch and self.save_steps > 1000:
            self.save_steps = self.save_steps % 1000

        if not os.path.isdir(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        if not self.redone:

            if not os.path.isdir(self.ckpt_resnet_path):
                os.makedirs(self.ckpt_resnet_path)

            #if not os.path.isdir(self.ckpt_rkn_encoder_path):
            #    os.makedirs(self.ckpt_rkn_encoder_path)

            #if not os.path.isdir(self.ckpt_rkn_score_path):
            #    os.makedirs(self.ckpt_rkn_score_path)

        if not os.path.isdir(self.train_path):
            os.makedirs(self.train_path)

        if not os.path.isdir(self.val_path):
            os.makedirs(self.val_path)

        self.current_epoch = tf.Variable(0)
        self.current_step = tf.Variable(0)
        if self.redone:
            if self.step_is_epoch:
                self.ckpt = tf.train.Checkpoint(epoch=self.current_epoch, optimizer=self.optimizer, net=self.model)
            else:
                self.ckpt = tf.train.Checkpoint(step=self.current_step, epoch=self.current_epoch, optimizer=self.optimizer, net=self.model)
        else:
            if self.step_is_epoch:
                self.ckpt = tf.train.Checkpoint(epoch=self.current_epoch, optimizer=self.optimizer)
            else:
                self.ckpt = tf.train.Checkpoint(step=self.current_step, epoch=self.current_epoch, optimizer=self.optimizer)

            self.ckpt_resnet = tf.train.Checkpoint(resnet=self.model.resnet)
            self.ckpt_resnet_manager = tf.train.CheckpointManager(self.ckpt_resnet, self.ckpt_resnet_path, max_to_keep=2)
            #self.ckpt_rkn_encoder = tf.train.Checkpoint(inference=self.model.rkn.inference_net, rkn=self.model.rkn.rkn_layer)
            #self.ckpt_rkn_encoder_manager = tf.train.CheckpointManager(self.ckpt_rkn_encoder, self.ckpt_rkn_encoder_path, max_to_keep=2)
            #self.ckpt_rkn_score = tf.train.Checkpoint(score=self.model.rkn.score_net)
            #self.ckpt_rkn_score_manager = tf.train.CheckpointManager(self.ckpt_rkn_score, self.ckpt_rkn_score_path, max_to_keep=2)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_path, max_to_keep=2)
        #self.load_pretrained(self.ckpt_resnet, self.ckpt_resnet_manager)
        if self.redone:
            self.load_redone()
        else:
            self.load()

    def forward_percent(self):
        print("forward percent")
        t_loss_mean = tf.keras.metrics.Mean(name='t_loss')
        t_acc_mean = tf.keras.metrics.Mean(name='t_acc')
        v_loss_mean = tf.keras.metrics.Mean(name='v_loss')
        v_acc_mean = tf.keras.metrics.Mean(name='v_acc')


        starting_epoch = int(self.ckpt.epoch)
        starting_step = int(self.ckpt.step)



        epoch_percent_train = self.train_size // 1000
        epoch_percent_train = 1 if epoch_percent_train == 0 else epoch_percent_train
        epoch_percent_val = self.val_size // 1000
        epoch_percent_val = 1 if epoch_percent_val == 0 else epoch_percent_val



        print("starting_step: ", starting_step)
        print("start progbar: ", starting_step // epoch_percent_train)

        print("train_size: ", self.train_size)
        print("val_size: ", self.val_size)

        for epoch in range(starting_epoch, self.epoch_max + 1):
            print("epoch : ", epoch)
            progbar = tf.keras.utils.Progbar(1000)
            progbar.update(starting_step//epoch_percent_train)

            self.lr = self.lr_fn(self.lr, epoch)
            self.optimizer.lr = self.lr

            # One epoch on TRAIN dataset
            #train_enum = self.train_ds.enumerate()
            #for element in train_enum.as_numpy_iterator():
            #print("train_ds: ", self.train_ds)
            for i, train_x in enumerate(self.train_ds, starting_step):
                t_loss_mean(self.model.compute_apply_gradients(train_x, self.optimizer))
                t_acc_mean(self.model.compute_accuracy(train_x))
                if i > (epoch_percent_train*1000):
                    break

                if i % epoch_percent_train == 0 and i != 0:
                    progbar.add(1)

                if i % (epoch_percent_train*50) == 0 and i != 0:
                    for val_x in self.val_ds.take(epoch_percent_val):
                        v_loss_mean(self.model.compute_loss(val_x))
                        v_acc_mean(self.model.compute_accuracy(val_x))


                    self.t_loss.append(t_loss_mean.result().numpy())
                    self.t_acc.append(t_acc_mean.result().numpy())
                    self.v_loss.append(v_loss_mean.result().numpy())
                    self.v_acc.append(v_acc_mean.result().numpy())

                    t_loss_mean.reset_states()
                    t_acc_mean.reset_states()
                    v_loss_mean.reset_states()
                    v_acc_mean.reset_states()

                if i != 0 and i % (epoch_percent_train*self.save_steps) == 0:

                    for val_x in self.val_ds.take(1):
                        if self.is_seq:
                            image_saver.KAST_View_Resnet(self.model, val_x, False,
                                                                             self.name + '_epoch_{:03d}_step_{:03d}_test'.format(
                                                                                 epoch, i // epoch_percent_train),
                                                                             self.img_path)
                            #image_saver.generate_gif_concat(self.model, val_x,
                            #                                self.name + '_epoch_{:03d}_test_gif'.format(epoch),
                            #                                self.img_path)
                        else:
                            image_saver.generate_and_save_images_compare_lab(self.model, val_x,
                                                                         self.name + '_epoch_{:03d}_step_{:03d}_test'.format(epoch, i//epoch_percent_train), self.img_path)
                    for train_x in self.train_ds.take(1):
                        if self.is_seq:
                            image_saver.KAST_View_Resnet(self.model, train_x, True,
                                                                             self.name + '_epoch_{:03d}_step_{:03d}_train'.format(
                                                                                 epoch, i // epoch_percent_train),
                                                                             self.img_path)
                        else:
                            image_saver.generate_and_save_images_compare_lab(self.model, train_x,
                                                                         self.name + '_epoch_{:03d}_step_{:03d}_train'.format(epoch, i//epoch_percent_train), self.img_path)
                    if not self.redone:
                        for test in self.test_ds.take(1):
                            image_saver.KAST_test(self.model, test, self.name + '_epoch_{:03d}_step_{:03d}_train'.format(epoch, i//epoch_percent_train), self.img_path)

                    print('i :', i)
                    print('epoch percent train: ', epoch_percent_train)
                    print('save step: ', self.save_steps)
                    x_axis = np.linspace(0, len(self.t_loss) / 1000, len(self.t_loss))
                    image_saver.curves([self.t_loss, self.v_loss], ['Training', 'Validation'],
                                       'training_validation_loss', self.img_path, 'Steps', 'Loss', x_axis)
                    image_saver.curves([self.t_acc, self.v_acc], ['Training', 'Validation'],
                                       'training_validation_accuracy', self.img_path, 'Steps', 'Accuracy', x_axis)
                    self.ckpt.step.assign(i+1)
                    if self.redone:
                        self.save_redone()
                    else:
                        self.save()

                    t_loss_mean.reset_states()
                    t_acc_mean.reset_states()
                    v_loss_mean.reset_states()
                    v_acc_mean.reset_states()

            starting_step = 0


            # Create temp image of loss


            #img_name = 'epoch_' + str(epoch)
            #for val_x in self.val_ds.take(1):
                #image_saver.compare_images(val_x, self.model.reconstruct(val_x), img_name, self.img_path)

            self.ckpt.epoch.assign_add(1)
        self.ckpt.step.assign(0)
        if self.redone:
            self.save_redone()
        else:
            self.save()

    def forward_epoch(self):
        print("forward epoch")
        t_loss_mean = tf.keras.metrics.Mean(name='t_loss')
        t_acc_mean = tf.keras.metrics.Mean(name='t_acc')
        v_loss_mean = tf.keras.metrics.Mean(name='v_loss')
        v_acc_mean = tf.keras.metrics.Mean(name='v_acc')

        starting_epoch = int(self.ckpt.epoch)

        print(starting_epoch)

        for epoch in range(starting_epoch, self.epoch_max + 1):
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
            image_saver.curves([self.t_loss, self.v_loss], ['Training', 'Validation'],
                               'training_validation_loss', self.img_path, 'Steps', 'Loss')
            image_saver.curves([self.t_acc, self.v_acc], ['Training', 'Validation'],
                               'training_validation_accuracy', self.img_path, 'Steps', 'Loss')

            img_name = 'epoch_' + str(epoch)
            for val_x in self.val_ds.take(1):
                if self.is_seq:
                    #image_saver.compare_images_seq(val_x, self.model.reconstruct(val_x), img_name, self.img_path)
                    image_saver.generate_and_save_images_compare_seq(self.model, val_x,
                                                                     self.name + '_epoch_{:03d}_test'.format(
                                                                         epoch),
                                                                     self.img_path)
                    image_saver.generate_gif_concat(self.model, val_x,
                                                    self.name + '_epoch_{:03d}_test_gif'.format(epoch),
                                                    self.img_path)
                else:
                    image_saver.compare_images(val_x, self.model.reconstruct(val_x), img_name, self.img_path)
            for train_x in self.train_ds.take(1):
                if self.is_seq:
                    #image_saver.compare_images_seq(val_x, self.model.reconstruct(val_x), img_name, self.img_path)
                    image_saver.generate_and_save_images_compare_seq(self.model, train_x,
                                                                     self.name + '_epoch_{:03d}_train'.format(
                                                                         epoch),
                                                                     self.img_path)
                else:
                    image_saver.compare_images(train_x, self.model.reconstruct(train_x), img_name, self.img_path)

            t_loss_mean.reset_states()
            t_acc_mean.reset_states()
            v_loss_mean.reset_states()
            v_acc_mean.reset_states()
            if epoch % self.save_steps == 0 or epoch == self.epoch_max:
                if self.redone:
                    self.save_redone()
                else:
                    self.save()
            self.ckpt.epoch.assign_add(1)

    def forward(self):
        if self.step_is_epoch:
            self.forward_epoch()
        else:
            self.forward_percent()

    """def forward(self, epochs=None, steps=None):
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

        starting_step = self.ckpt.step
        starting_epoch = self.ckpt.epoch

        steps_in_epochs = np.float32(steps)/np.float32(self.train_size)

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
            self.ckpt.epoch.assign_add(1)

        for step in range(starting_step, steps + 1):
            progbar = tf.keras.utils.Progbar(100)
            self.optimizer.lr = self.lr

            for i, train_x in enumerate(self.train_ds.take(steps)):
                t_loss_mean(self.model.compute_apply_gradients(train_x, self.optimizer))
                t_acc_mean(self.model.compute_accuracy(train_x))
                if i % (steps // 100):
                    progbar.add(1)

            for i, val_x in enumerate(self.val_ds):
                v_loss_mean(self.model.compute_loss(val_x))
                v_acc_mean(self.model.compute_accuracy(val_x))"""

    def save_redone(self):
        # Save info about loss and acuracy of training and validation dataset
        t_loss_path = os.path.join(self.train_path, 'loss.npy')
        t_acc_path = os.path.join(self.train_path, 'accuracy.npy')
        v_loss_path = os.path.join(self.val_path, 'loss.npy')
        v_acc_path = os.path.join(self.val_path, 'accuracy.npy')

        save_path = self.ckpt_manager.save()

        np.save(t_loss_path, np.asarray(self.t_loss))
        np.save(t_acc_path, np.asarray(self.t_acc))
        np.save(v_loss_path, np.asarray(self.v_loss))
        np.save(v_acc_path, np.asarray(self.v_acc))

        print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path))

    def save(self):
        #Save info about loss and acuracy of training and validation dataset
        t_loss_path = os.path.join(self.train_path, 'loss.npy')
        t_acc_path = os.path.join(self.train_path, 'accuracy.npy')
        v_loss_path = os.path.join(self.val_path, 'loss.npy')
        v_acc_path = os.path.join(self.val_path, 'accuracy.npy')

        #Save model with checkpoint
        print("self.ckpt.step", self.ckpt.step)
        print("self.ckpt.epoch", self.ckpt.epoch)
        print("self.ckpt_path", self.ckpt_path)

        save_path = self.ckpt_manager.save()
        save_path_resnet = self.ckpt_resnet_manager.save()
        #save_path_rkn_score = self.ckpt_rkn_score_manager.save()
        #save_path_rkn_encoder = self.ckpt_rkn_encoder_manager.save()
        np.save(t_loss_path, np.asarray(self.t_loss))
        np.save(t_acc_path, np.asarray(self.t_acc))
        np.save(v_loss_path, np.asarray(self.v_loss))
        np.save(v_acc_path, np.asarray(self.v_acc))
        if self.step_is_epoch:
            print("Saved checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path))
            print("Saved Resnet checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path_resnet))
            #print("Saved RKN encoder checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path_rkn_encoder))
            #print("Saved RKN score checkpoint for epoch {}: {}".format(int(self.ckpt.epoch), save_path_rkn_score))
        else:
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            print("Saved Resnet checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_resnet))
            #print("Saved RKN encoder checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_rkn_encoder))
            #print("Saved RKN score checkpoint for step {}: {}".format(int(self.ckpt.step), save_path_rkn_score))


    def load_redone(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
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

    def load_pretrained(self, ckpt, ckpt_manager):
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        if ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
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

    def load(self):
        ## RESNET
        self.ckpt_resnet.restore(self.ckpt_resnet_manager.latest_checkpoint).expect_partial()
        if self.ckpt_resnet_manager.latest_checkpoint:
            print("Restored Resnet from {}".format(self.ckpt_resnet_manager.latest_checkpoint))
        else:
            print("Initializing Resnet from scratch.")

        ## RKN ENCODER
        #self.ckpt_rkn_encoder.restore(self.ckpt_rkn_encoder_manager.latest_checkpoint).expect_partial()
        #if self.ckpt_rkn_encoder_manager.latest_checkpoint:
        #    print("Restored RKN encoder from {}".format(self.ckpt_rkn_encoder_manager.latest_checkpoint))
        #else:
        #    print("Initializing RKN encoder from scratch.")

        ## RKN SCORE
        #self.ckpt_rkn_score.restore(self.ckpt_rkn_score_manager.latest_checkpoint).expect_partial()
        #if self.ckpt_rkn_score_manager.latest_checkpoint:
        #    print("Restored RKN score from {}".format(self.ckpt_rkn_score_manager.latest_checkpoint))
        #else:
        #    print("Initializing RKN score from scratch.")


        ## TRAINING
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
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












