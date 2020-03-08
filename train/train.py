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


def train(ds, model, lr, epochs, batch_size, ckpt_epoch, directory_path, directory_name='default_filename', img_while_training=True):

    optimizer = tf.keras.optimizers.Adam(lr)
    main_path = os.path.join(directory_path, directory_name)
    ckpt_path = os.path.join(main_path, 'ckpts')
    training_data_path = os.path.join(main_path, 'training')
    validation_data_path = os.path.join(main_path, 'validation')
    images_path = os.path.join(main_path, 'imgs')
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.isdir(training_data_path):
        os.makedirs(training_data_path)
    if not os.path.isdir(validation_data_path):
        os.makedirs(validation_data_path)
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    training_loss_npy_path = os.path.join(training_data_path, 'loss.npy')
    training_acc_npy_path = os.path.join(training_data_path, 'accuracy.npy')
    validation_loss_npy_path = os.path.join(validation_data_path, 'loss.npy')
    validation_acc_npy_path = os.path.join(validation_data_path, 'accuracy.npy')

    training_loss_np = np.load(training_loss_npy_path) if os.path.isfile(training_loss_npy_path) else np.zeros(epochs)
    training_acc_np = np.load(training_acc_npy_path) if os.path.isfile(training_acc_npy_path) else np.zeros(epochs)
    validation_loss_np = np.load(validation_loss_npy_path) if os.path.isfile(validation_loss_npy_path) else np.zeros(epochs)
    validation_acc_np = np.load(validation_acc_npy_path) if os.path.isfile(validation_acc_npy_path) else np.zeros(epochs)

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

    #train_loss_results = []
    #test_loss_results = []
    #train_accuracy_results = []
    #test_accuracy_results = []

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
                        image_saver.generate_and_save_images_compare_lab(model, test_x, directory_name+'epoch_{:03d}'.format(epoch)+'_step_'+str(i), path=images_path)


        end_time = time.time()

        if epoch == 30:
            lr = lr*0.1
            optimizer.lr = lr

        if epoch == 50:
            lr = lr*0.1
            optimizer.lr = lr

        if epoch % 1 == 0:
            #### See if not only take(1)
            for test_x in test_dataset:
                test_loss_mean(model.compute_loss(test_x))
                test_accuracy_mean(model.compute_accuracy(test_x))


            elbo = -test_loss_mean.result()
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            logging.info('Loss_train_{:.6f}_test_{:.6f}'.format(-train_loss_mean.result(), -test_loss_mean.result()))
            #train_loss_results.append(train_loss_mean.result())
            #test_loss_results.append(test_loss_mean.result())
            #train_accuracy_results.append(train_accuracy_mean.result())
            #test_accuracy_results.append(test_accuracy_mean.result())
            training_loss_np[epoch-1] = train_loss_mean.result().numpy()
            training_acc_np[epoch-1] = train_accuracy_mean.result().numpy()
            validation_loss_np[epoch - 1] = test_loss_mean.result().numpy()
            validation_acc_np[epoch - 1] = test_accuracy_mean.result().numpy()

            train_loss_mean.reset_states()
            test_loss_mean.reset_states()
            train_accuracy_mean.reset_states()
            test_accuracy_mean.reset_states()
            if img_while_training:
                image_saver.img_loss_accuracy(training_loss_np, validation_loss_np, training_acc_np, validation_acc_np, filename='loss_accuracy_Lab_temp', path=images_path)
                for test_x in test_dataset.take(1):
                    image_saver.generate_and_save_images_compare_lab(model, test_x, directory_name+'_epoch_{:03d}_test'.format(epoch), images_path)
                    means, logvar = model.encode(test_x)
                    var = tf.exp(logvar)
                    image_saver.points([means[0, :]], ['mean'], 'epoch_{:03d}_mean_test'.format(epoch), images_path)
                    image_saver.points([var[0, :]], ['var'], 'epoch_{:03d}_var_test'.format(epoch), images_path)
                for train_x in train_dataset.take(1):
                    image_saver.generate_and_save_images_compare_lab(model, train_x, directory_name+'_epoch_{:03d}_train'.format(epoch), images_path)
                    means, logvar = model.encode(train_x)
                    var = tf.exp(logvar)
                    image_saver.points([means[0, :]], ['mean'], 'epoch_{:03d}_mean_tran'.format(epoch), images_path)
                    image_saver.points([var[0, :]], ['var'], 'epoch_{:03d}_var_train'.format(epoch), images_path)

        ckpt.step.assign_add(1)
        if int(ckpt.step) % ckpt_epoch == 0 or epoch == epochs:
            print("ckpt.step :", int(ckpt.step))
            print("epoch :", epoch)
            save_path = manager.save()
            np.save(training_loss_npy_path, training_loss_np)
            np.save(training_acc_npy_path, training_acc_np)
            np.save(validation_loss_npy_path, validation_loss_np)
            np.save(validation_acc_npy_path, validation_acc_np)
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(-test_loss_mean.result()))
    if img_while_training:
        image_saver.img_loss_accuracy(training_loss_np, validation_loss_np, training_acc_np, validation_acc_np, filename="loss_accuracyLab", path=images_path)

    return training_loss_np, validation_loss_np, training_acc_np, validation_acc_np

def multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lrs, epochs, batch_size, ckpt_epochs, directory_name, path, models_std, legends=None):

    model_args = [datasets, models_type, models_arch, models_latent_space, models_use_bn, models_std, lrs, epochs, batch_size, ckpt_epochs]
    max_len = max(map(len, model_args))
    print(max_len)

    path_directory = os.path.join(path, directory_name)
    if not os.path.isdir(path_directory):
        os.makedirs(path_directory)

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
    models = []

    # Construct the model
    for i in range(max_len):
        #Get name
        str_ds = model_args[0][i]
        str_model = model_args[1][i]
        str_arch = '_'.join(str(x) for x in model_args[2][i])
        str_lat = 'lat' + str(model_args[3][i])
        str_use_bn = 'BN' if model_args[4][i] else ''
        #str_std = 'std' + str(model_args[5][i])
        #str_all = '_'.join(filter(None, [str_ds, str_model, str_arch, str_lat, str_std, str_use_bn]))
        str_all = '_'.join(filter(None, [str_ds, str_model, str_arch, str_lat, str_use_bn]))

        #Construct the model
        dataset = ds[model_args[0][i]][0]
        shape = ds[model_args[0][i]][1]
        model_type = model_args[1][i]
        model_arch = model_args[2][i].copy()
        model_lat = model_args[3][i]
        model_use_bn = model_args[4][i]
        #model_std = model_args[5][i]
        lr = model_args[6][i]
        epoch = model_args[7][i]
        bs = model_args[8][i]
        ckpt_epoch = model_args[9][i]

        print(model_lat)

        model = construct_model.get_model(model_type, model_arch, model_lat, shape, model_use_bn)
        models.append(model)

        #Train
        print(dataset)
        print(model)
        print(lr)
        print(epoch)
        print(bs)
        print(ckpt_epoch)


        train_l, test_l, train_acc, test_acc = train(dataset, model, lr, epoch, bs, ckpt_epoch, path_directory,  str_all, True)

        #Get curves and output them
        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_losses.append(test_l)
        test_accs.append(test_acc)




        legendes.append(str_all)

    if legends is None:
        legends = legendes

    image_saver.curves(test_accs, legends, directory_name+'_curves', path_directory, 'epochs', 'accuracy (L2)')
    dataset_test = ds[model_args[0][0]][0][1]

    images = []
    for test in dataset_test.batch(4).take(1):
        print(test.shape)
        ground_truth = test.numpy()
        images.append(ground_truth)
        for model in models:
            output = model.reconstruct(test)
            images.append(output.numpy())


    image_saver.compare_multiple_images(images, directory_name+'_images', path_directory)



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



datasets = ['cifar10Lab']
models_type = ['CVAE']  # or ['AE']
models_arch = [[128, 256, 512]]
#models_arch = [[64, 128, 256]]
models_latent_space = [1024]
models_use_bn = [False, True]
lr = [1e-4]
epochs = [40]
batch_size = [128]
my_drive_path = '/content/drive/My Drive/Colab Data/AE/'
#directories_name = ['ckpts_aeLab_128x256x512_lat1024', 'ckpts_aeLab_128x256x512_lat1024_BN']
ckpt_epoch = [10]
filename = 'batch_normalization'



multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lr, epochs, batch_size, ckpt_epoch, directory_name, my_drive_path)
"""


datasets = ['cifar10Lab']
models_type = ['CVAE']  # or ['AE']
models_arch = [[128, 256, 512]]
models_std = [0.05]
#models_arch = [[64, 128, 256]]
#models_latent_space = [64]
models_latent_space = [512, 1024, 2048, 4096]
#models_latent_space = [128, 256, 512, 1024, 2048, 4096]
models_use_bn = [False]
lr = [1e-4]
epochs = [70]
batch_size = [128]
legends = ['512', '1024', '2048', '4096']
my_drive_path = '/content/drive/My Drive/Colab Data/AE/'
#ckpt_path = ['ckpts_aeLab_128x256x512_lat1024', 'ckpts_sbaeLab_128x256x512_lat1024', 'ckpts_aeLab_256x512x1024_lat2048', 'ckpts_sbaeLab_256x512x1024_lat2048']
ckpt_epoch = [20]
directory_name = 'VAE_COMPARE_Latent_Space'



multitraining(datasets, models_type, models_arch, models_latent_space, models_use_bn, lr, epochs, batch_size, ckpt_epoch, directory_name, my_drive_path, models_std, legends)


#res = [[0.00789244, 0.0055954787], [0.007047541, 0.004881351], [0.0083873095, 0.0067818933]]
#legende = ['cifar10Lab_AE_256_128_64_lat256', 'cifar10Lab_AE_512_256_128_lat512_BN', 'cifar10Lab_AE_1024_512_256_128_lat1024']
#image_saver.curves(res, legende, 'truc')

#print(len(models_arch))


#train_l, test_l, train_acc, test_acc = train(ds, model, lr=1e-4, epochs=40, batch_size=128, ckpt_path='./ckpts_sbaeLab', ckpt_epoch=10)

#ll_train_l.append(train_l)



#logging.basicConfig(filename='./training_Lab.log', level=logging.DEBUG)

#train(ds, model, lr=1e-4, epochs=40, batch_size=128, ckpt_path='./ckpts_sbaeLab', ckpt_epoch=10)

