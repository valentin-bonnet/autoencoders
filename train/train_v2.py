import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from models.vae import VAE
from models.cvae import CVAE
import utils.image_saver
import datas.cirafLoader
from datas import DAVISLoad

AUTOTUNE = tf.data.experimental.AUTOTUNE


def train(model_type='CVAE', number_layers=4, latent_dim=1024, epochs=30):

    if model_type == 'CVAE':
        model = CVAE(number_layers, latent_dim)
    elif model_type == 'VAE':
        model = VAE(number_layers, latent_dim)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
    test_loss_mean = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy_mean = tf.keras.metrics.Mean(name='train_accuracy')
    test_accuracy_mean = tf.keras.metrics.Mean(name='test_accuracy')

    train_dataset = DAVISLoad.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS', 'train')
    test_dataset = DAVISLoad.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS', 'val')
    #train_dataset, test_dataset = datas.cirafLoader.cirafloader()
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(128).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.shuffle(buffer_size=10000).batch(128).prefetch(buffer_size=AUTOTUNE)





    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    #random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])



    # generate_and_save_images(model, 0, random_vector_for_generation)
    train_loss_results = []
    test_loss_results = []
    train_accuracy_results = []
    test_accuracy_results = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_loss_mean(model.compute_apply_gradients(train_x, optimizer))
            train_accuracy_mean(model.compute_accuracy(train_x))

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
            train_loss_results.append(train_loss_mean.result())
            test_loss_results.append(test_loss_mean.result())
            train_accuracy_results.append(train_accuracy_mean.result())
            test_accuracy_results.append(test_accuracy_mean.result())
            train_loss_mean.reset_states()
            test_loss_mean.reset_states()
            train_accuracy_mean.reset_states()
            test_accuracy_mean.reset_states()
            for test_x in test_dataset.take(1):
                utils.image_saver.generate_and_save_images_compare(
                    model, epoch, test_x)

    utils.image_saver.img_loss_accuracy(epochs, train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename="DAVIS_big_loss")


train()