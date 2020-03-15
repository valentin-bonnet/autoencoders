import sys
import os
curr_dir = os.getcwd()
sys.path.append(curr_dir)


import cifarLoader
import DAVISLoader
import ImagenetResizedLoader


"""
** TO DO:
** - choose preprocess
** - file location
"""
def get_dataset(dataset):
    if dataset == 'DAVIS':
        train_ds = DAVISLoader.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                                              'train')
        test_ds = DAVISLoader.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                                             'val')
        shape = 32
    elif dataset == 'cifar10':
        train_ds, test_ds = cifarLoader.cifarloader()
        shape = 32

    elif dataset == 'cifar10Lab':
        train_ds, test_ds = cifarLoader.cifarloaderLab()
        shape = 32

    elif dataset == 'imagenetresized32':
        train_ds, test_ds = ImagenetResizedLoader.imagenetresized32loaderLab()
        shape = 32

    elif dataset == 'imagenetresized64':
        train_ds, test_ds = ImagenetResizedLoader.imagenetresized64loaderLab()
        shape = 64

    else:
        print("No good dataset selected")

    return (train_ds, test_ds), shape