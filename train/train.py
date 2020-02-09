import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
curr_dir = os.getcwd()
path = os.path.abspath(os.path.join(curr_dir, '..'))
sys.path.append(os.path.join(path, 'models'))
sys.path.append(os.path.join(path, 'utils'))
sys.path.append(os.path.join(path, 'datas'))

import image_saver
import datasetLoader
import construct_model

import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE


def train(ds, model, lr, epochs, batch_size, ckpt_path, ckpt_epoch):

    optimizer = tf.keras.optimizers.Adam(lr)

    current_epoch = tf.Variable(1)
    ckpt = tf.train.Checkpoint(step=current_epoch, optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=epochs/ckpt_epoch)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
    test_loss_mean = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy_mean = tf.keras.metrics.Mean(name='train_accuracy')
    test_accuracy_mean = tf.keras.metrics.Mean(name='test_accuracy')

    train_dataset, test_dataset = ds
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)





    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    #random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])



    # generate_and_save_images(model, 0, random_vector_for_generation)
    train_loss_results = []
    test_loss_results = []
    train_accuracy_results = []
    test_accuracy_results = []

    starting_epoch = current_epoch.numpy()
    len_train = tf.data.experimental.cardinality(train_dataset).numpy()
    for epoch in range(starting_epoch, epochs + 1):
        start_time = time.time()
        progbar = tf.keras.utils.Progbar(len_train)
        for i, train_x in enumerate(train_dataset):
            progbar.update(i+1)
            train_loss_mean(model.compute_apply_gradients(train_x, optimizer))
            train_accuracy_mean(model.compute_accuracy(train_x))
            if i % (len_train/10) == 0:
                for test_x in test_dataset.take(1):
                    image_saver.generate_and_save_images_compare_lab(model, epoch, test_x, 'SBAE_Lab_step_'+str(i))


        end_time = time.time()

        if epoch % 1 == 0:

            for test_x in test_dataset:
                test_loss_mean(model.compute_loss(test_x))
                test_accuracy_mean(model.compute_accuracy(test_x))


            elbo = -test_loss_mean.result()
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            logging.info('Loss_train_{:.6f}_test_{:.6f}'.format(-train_loss_mean.result(), -test_loss_mean.result()))
            train_loss_results.append(train_loss_mean.result())
            test_loss_results.append(test_loss_mean.result())
            train_accuracy_results.append(train_accuracy_mean.result())
            test_accuracy_results.append(test_accuracy_mean.result())
            train_loss_mean.reset_states()
            test_loss_mean.reset_states()
            train_accuracy_mean.reset_states()
            test_accuracy_mean.reset_states()
            image_saver.img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename="loss_accuracy_Lab_temp")
            for test_x in test_dataset.take(1):
                image_saver.generate_and_save_images_compare_lab(
                    model, epoch, test_x, 'SBAE_Lab')
        ckpt.step.assign_add(1)
        if int(ckpt.step) % ckpt_epoch == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(-test_loss_mean.result()))

    image_saver.img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename="loss_accuracyLab")

# Dataset


logging.basicConfig(filename='./training_Lab.log', level=logging.DEBUG)

ds = datasetLoader.get_dataset('cifar10Lab')
model = construct_model.get_model('SBAE', [64, 128, 256], 1024, 32)

train(ds, model, lr=1e-4, epochs=40, batch_size=128, ckpt_path='./ckpts_sbaeLab', ckpt_epoch=10)
