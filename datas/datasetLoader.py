import datas.cifarLoader
import datas.DAVISLoader
import datas.ImagenetResizedLoader


"""
** TO DO:
** - choose preprocess
** - file location
"""
def get_dataset(dataset):
    if dataset == 'DAVIS':
        train_ds = datas.DAVISLoader.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                                              'train')
        test_ds = datas.DAVISLoader.get_dataset('/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS',
                                             'val')
    elif dataset == 'cifar10':
        train_ds, test_ds = datas.cifarLoader.cifarloader()

    elif dataset == 'cifar10Lab':
        train_ds, test_ds = datas.cifarLoader.cifarloaderLab()

    elif dataset == 'imagenetresized64':
        train_ds, test_ds = datas.ImagenetResizedLoader.imagenetresized64loaderLab()

    else:
        print("No good dataset selected")

    return (train_ds, test_ds)