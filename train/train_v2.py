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
import Multitraining
import Dataset

import logging

AUTOTUNE = tf.data.experimental.AUTOTUNE


def train(ds, model, lr, epochs, batch_size, ckpt_path, ckpt_epoch, filename='default_filename', path='./', img_while_training=True):

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
            if img_while_training:
                if i % (len_train/10) == 0:
                    for test_x in test_dataset.take(1):
                        image_saver.generate_and_save_images_compare_lab(model, epoch, test_x, 'temp_'+filename+'_step_'+str(i), path=path+ckpt_path+'/imgs/')


        end_time = time.time()

        if epoch % 30 == 0:
            lr = lr*0.1
            optimizer.lr = lr

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
            if img_while_training:
                image_saver.img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename='loss_accuracy_Lab_temp', path=path+ckpt_path+'/imgs/')
                for test_x in test_dataset.take(1):
                    image_saver.generate_and_save_images_compare_lab(model, epoch, test_x, 'temp_'+filename, path+ckpt_path+'/imgs/')
        ckpt.step.assign_add(1)
        if int(ckpt.step) % ckpt_epoch == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(-test_loss_mean.result()))
    if img_while_training:
        image_saver.img_loss_accuracy(train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results, filename="loss_accuracyLab", path=path+ckpt_path+'/imgs/')
    return train_loss_results, test_loss_results, train_accuracy_results, test_accuracy_results

def multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lrs, epochs, batch_size, ckpt_paths, ckpt_epochs, filename, path):

    model_args = [datasets, models_type, models_arch, models_latent_space, models_use_bn, lrs, epochs, batch_size, ckpt_paths, ckpt_epochs]
    max_len = max(map(len, model_args))
    print(max_len)

    #Adapt to get all parameters to same size
    for i, arg in enumerate(model_args):
        temp_arg = arg.copy()
        for j in range((max_len-1)//len(arg)):
            arg.extend(temp_arg)
        model_args[i] = arg[:max_len]

    #Use this to get unique dataset used multiple times
    ds = {}
    for dataset in datasets:
        if dataset not in ds:
            train_test_ds, shape = datasetLoader.get_dataset(dataset)
            ds.update({dataset: [train_test_ds, shape]})

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    legendes = []
    # Construct the model,
    for i in range(max_len):
        #Get name
        str_ds = model_args[0][i]
        str_model = model_args[1][i]
        str_arch = '_'.join(str(x) for x in model_args[2][i])
        str_lat = 'lat' + str(model_args[3][i])
        str_use_bn = 'BN' if model_args[4][i] else ''
        str_all = '_'.join(filter(None, [str_ds, str_model, str_arch, str_lat, str_use_bn]))

        #Construct the model
        dataset = ds[model_args[0][i]][0]
        shape = ds[model_args[0][i]][1]
        model_type = model_args[1][i]
        model_arch = model_args[2][i]
        model_lat = model_args[3][i]
        model_use_bn = model_args[4][i]

        lr = model_args[5][i]
        epoch = model_args[6][i]
        bs = model_args[7][i]
        ckpt_path = model_args[8][i]
        ckpt_epoch = model_args[9][i]

        print(model_lat)

        model = construct_model.get_model(model_type, model_arch, model_lat, shape, model_use_bn)

        #Train
        print(dataset)
        print(model)
        print(lr)
        print(epoch)
        print(bs)
        print(ckpt_path)
        print(ckpt_epoch)

        train_l, test_l, train_acc, test_acc = train(dataset, model, lr, epoch, bs, ckpt_path, ckpt_epoch, str_all, path)

        #Get curves and output them
        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_losses.append(test_l)
        test_accs.append(test_acc)



        legendes.append(str_all)

    image_saver.curves(test_accs, legendes, filename, path)




"""
    print(model_args)
    for i in range(max_len):
        str_ds = model_args[0][i]
        str_model = model_args[1][i]
        str_arch = '_'.join(str(x) for x in model_args[2][i])
        str_lat = 'lat'+str(model_args[3][i])
        str_use_bn = 'BN' if model_args[4][i] else ''
        str_all = '_'.join(filter(None, [str_ds, str_model, str_arch, str_lat, str_use_bn]))

        print(str_all)
        test_data = [[10, 7, 5, 6, 4, 3, 3.1, 2.8], [1, 2, 4]]
        image_saver.curves(test_data, ['test_1', 'test_2'], ['test_1_img', 'test_2_img'])

"""



    #for model in models:
    #    train(ds, model, lr, epochs, batch_size, ckpt_path, ckpt_epoch)


# Dataset


#ds = datasetLoader.get_dataset('cifar10Lab')
#datasets = ['cifar10Lab']
#model = construct_model.get_model('AE', [64, 128, 256], 1024, 32)
"""
datasets = ['cifar10Lab']
models_type = ['AE']  # or ['AE']
models_arch = [[32, 64, 128], [32, 64, 128, 128], [128, 256, 512], [128, 256, 512, 512], [512, 1024, 2048], [512, 1024, 2048, 2048]]
#models_arch = [[64, 128, 256]]
models_latent_space = [1024]
models_use_bn = [False]
lr = [1e-4]
epochs = [40]
batch_size = [128]
my_drive_path = '/content/drive/My Drive/Colab Data/AE/'
ckpt_path = ['ckpts_aeLab_32x64x128', 'ckpts_aeLab_32x64x128x128', 'ckpts_aeLab_128x256x512', 'ckpts_aeLab_128x256x512x512', 'ckpts_aeLab_512x1024x2048', 'ckpts_aeLab_512x1024x2048x2048']
ckpt_epoch = [10]
filename = 'models_layers'

multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lr, epochs, batch_size, ckpt_path, ckpt_epoch, filename, my_drive_path)
"""


datasets = ['cifar10Lab']
models_type = ['AE']  # or ['AE']
models_arch = [[128, 256, 512]]
#models_arch = [[64, 128, 256]]
models_latent_space = [1024]
models_use_bn = [False, True]
lr = [1e-4]
epochs = [40]
batch_size = [128]
my_drive_path = '/content/drive/My Drive/Colab Data/AE/'
ckpt_path = ['ckpts_aeLab_128x256x512_lat1024', 'ckpts_aeLab_128x256x512_lat1024_BN']
ckpt_epoch = [10]
filename = 'batch_normalization'



#multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lr, epochs, batch_size, ckpt_path, ckpt_epoch, filename, my_drive_path)



#datasets = ['cifar10Lab']
#models_type = ['AE', 'SBAE', 'AE', 'SBAE']  # or ['AE']
#models_arch = [[128, 256, 512], [128, 256, 512], [256, 512, 1024], [256, 512, 1024]]
#models_arch = [[64, 128, 256]]
#models_latent_space = [1024, 1024, 2048, 2048]
#models_use_bn = [False]
#lr = [1e-4]
#epochs = [40]
#batch_size = [128]
#my_drive_path = '/content/drive/My Drive/Colab Data/AE/'
#ckpt_path = ['ckpts_aeLab_128x256x512_lat1024', 'ckpts_sbaeLab_128x256x512_lat1024', 'ckpts_aeLab_256x512x1024_lat2048', 'ckpts_sbaeLab_256x512x1024_lat2048']
#ckpt_epoch = [10]
#filename = 'ae_sbae'

KAST = True
if KAST:
    datasets = Dataset.Dataset('oxuva')

    model1 = construct_model.get_model('KAST')


    models = [model1]
    lrs = [3e-4]
    optimizers = [tf.keras.optimizers.Adam(lr) for lr in lrs]
    def lr_fn(lr, step):
        if step == 4 or step == 6 or step == 8 or step == 10 or step == 12 or step == 14 or step == 16 or step == 18:
            return lr*0.5
        else:
            return lr
    lrs_fn = [lr_fn]
    batch_size = 4
    epochs_max = [20]
    saves_epochs = [100]
    #directory_path = './content/drive/My Drive/Colab Data/AE/'
    directory_path = '/content/drive/My Drive/Colab Data/AE/'
    path_to_directory = directory_path+'KAST_Local_Memory'
    step_is_epoch = False
    multi = Multitraining.Multitraining(datasets, batch_size, models, optimizers, lrs, lrs_fn, epochs_max, saves_epochs, path_to_directory, step_is_epoch)
    print("Multitraining Done")
    multi.forward()

else:
    datasets = Dataset.Dataset('imagenetresized64')
    #datasets = Dataset.Dataset('cifar10')

    model1 = construct_model.get_model('SBAE', layers=[[64, 7, 2], [64, 3, 1], [128, 3, 1], [128, 3, 2], [256, 3, 1], [256, 3, 1], [256, 3, -2], [256, 3, 1]])
    #model128 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=128)
    #model256 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=256)
    #model512 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=512)
    #model1024 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=1024)
    #model2048 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=2048)
    #model4096 = construct_model.get_model('AE', layers=[128, 256, 512], latent_dim=4096)

    models = [model1]
    lrs = [3e-4]
    optimizers = [tf.keras.optimizers.Adam(lr) for lr in lrs]
    def lr_fn(lr, step):
        if step == 10 or step == 15:
            return lr*0.1
        else:
            return lr
    lrs_fn = [lr_fn]
    batch_size = 1024
    epochs_max = [70]
    saves_epochs = [10]
    #directory_path = './content/drive/My Drive/Colab Data/AE/'
    directory_path = '/content/drive/My Drive/Colab Data/AE/'
    path_to_directory = directory_path+'AE_redo'
    #step_is_epoch = False
    step_is_epoch = True
    multi = Multitraining.Multitraining(datasets, batch_size, models, optimizers, lrs, lrs_fn, epochs_max, saves_epochs, path_to_directory, step_is_epoch)
    print("Multitraining Done")
    multi.forward()
"""
#multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lr, epochs, batch_size, ckpt_path, ckpt_epoch, filename, my_drive_path)


#res = [[0.00789244, 0.0055954787], [0.007047541, 0.004881351], [0.0083873095, 0.0067818933]]
#legende = ['cifar10Lab_AE_256_128_64_lat256', 'cifar10Lab_AE_512_256_128_lat512_BN', 'cifar10Lab_AE_1024_512_256_128_lat1024']
#image_saver.curves(res, legende, 'truc')

#print(len(models_arch))


#train_l, test_l, train_acc, test_acc = train(ds, model, lr=1e-4, epochs=40, batch_size=128, ckpt_path='./ckpts_sbaeLab', ckpt_epoch=10)

#ll_train_l.append(train_l)



#logging.basicConfig(filename='./training_Lab.log', level=logging.DEBUG)

#train(ds, model, lr=1e-4, epochs=40, batch_size=128, ckpt_path='./ckpts_sbaeLab', ckpt_epoch=10)

"""