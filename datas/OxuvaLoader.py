import tensorflow as tf
import oxuvaTFRecord
from skimage import color
import numpy as np

_imgs_per_folder = [4172, 2552, 1442, 10862, 3182, 1442, 2342, 4142, 3632, 1352, 1382, 1412, 8132, 1892, 1472, 1892, 1442, 4082, 2342, 9992, 1112, 3902, 3182, 1442, 37442, 4952, 1442, 4142, 2792, 1442, 3692, 6842, 13592, 2192, 8702, 1442, 12512, 5222, 1382, 1442, 10892, 1052, 9032, 12692, 9482, 4142, 992, 13082, 1322, 1262, 1052, 8642, 1982, 6392, 3242, 2342, 1382, 1892, 1052, 11792, 8912, 1892, 6842, 12212, 7741, 6842, 4982, 3182, 3692, 5972, 6392, 1442, 2732, 15812, 3242, 7292, 5372, 1412, 1232, 14492, 2702, 3152, 1532, 3182, 10892, 5492, 2702, 1742, 1442, 2672, 1292, 5492, 15842, 5882, 1382, 1382, 1232, 3242, 2582, 1442, 1382, 1442, 16592, 1382, 4082, 3242, 1382, 1382, 2342, 5312, 1262, 2342, 8192, 2552, 4142, 1352, 1292, 2282, 1832, 4592, 1442, 3902, 1382, 5852, 10082, 2792, 18092, 1442, 19052, 2342, 1442, 1172, 7322, 2342, 2192, 4082, 1382, 1442, 1382, 1382, 2342, 1442, 1382, 9032, 3062, 1382, 1442, 3182, 2372, 2432, 2732, 2342, 1352, 5492, 1412, 1442, 5432, 5042, 3632, 13532, 3122, 1892, 1442, 2792, 1442, 4142, 1382, 1292, 4142, 2282, 1442, 1442, 1322, 2252, 1352, 1892, 1082, 1382, 5042, 1952, 29342, 1802, 1442, 4442, 1442, 13982, 5492, 2732, 13142, 5012, 5942, 1442, 1442, 1382, 5942, 1382, 1442, 1502, 11282, 3242, 9992, 4982, 1442, 1742, 1442, 2342, 1412, 7172, 4592, 7742, 1442, 4142, 3182, 4142, 1442, 5432, 2132, 1232, 2792, 13112, 3182, 3032, 9542, 9542, 1442, 9032, 2762, 2282, 3902, 1442, 3122, 3692, 1442, 4082, 1472, 1532, 3632, 1322, 9482, 12602, 4142, 1352, 1472, 1442, 1442, 1532, 2792, 3242, 4622, 1292, 1442, 4742, 18541, 1352, 1382, 1382, 1382, 3242, 2282, 2342, 2612, 5942, 9902, 4112, 1622, 3692, 6392, 1442, 1442, 1772, 1442, 1382, 1442, 3242, 5042, 2342, 1172, 2852, 10952, 2282, 2282, 2282, 9032, 5492, 12182, 8132, 1442, 4142, 4142, 1442, 4982, 1442, 1442, 2792, 9032, 1292, 4532, 1442, 1202, 7292, 1502, 1322, 1292, 2342, 2792, 1352, 4142, 1382, 1382, 3332, 5942, 2762, 1862, 5432, 1892, 15392, 10832, 2222, 2762, 1442, 3242, 1352, 902, 1412, 3242, 1592, 1892, 11702, 4592, 1382, 1352, 1382, 2762, 1442, 2732, 1382, 1382]


def _rgb2lab(image):
    return color.rgb2lab(image)

def _preprocess_one_ds(parsed_data):
    img = parsed_data['image_raw']
    img = tf.image.decode_jpeg(img)
    img_lab = tf.py_function(func=_rgb2lab, inp=[img], Tout=tf.float32)
    img_lab = tf.image.convert_image_dtype(img_lab, tf.float32)
    img_normalized = img_lab + [0., 128.0, 128.0]
    img_normalized = (img_normalized / [50.0, 127.5, 127.5]) - 1.0
    return img_normalized

def oxuva_loader(path='/content/drive/My Drive/Colab Data/Datasets/oxuva_256/', seq_size=8):
    datasets = oxuvaTFRecord.tfrecord_to_dataset(path)
    i = 1
    np_size = np.array(_imgs_per_folder)
    nb_batch = np_size // seq_size
    all_size = np.sum(nb_batch)

    for data in datasets:
        ds = data.map(_preprocess_one_ds)
        ds = ds.batch(seq_size, drop_remainder=True)
        if i == 1:
            oxuva = ds
        else:
            oxuva = oxuva.concatenate(ds)
        i = i +1


    val_size = 10
    train_size = all_size - val_size
    for train in oxuva.take(1):
        print("OXUVA !")
        print(train.shape)
    oxuva_test = oxuva.take(val_size)
    oxuva_train = oxuva.skip(val_size)

    for train in oxuva_test.take(1):
        print("OXUVA 2 !")
        print(train.shape)

    for train in oxuva_train.take(1):
        print("OXUVA 3 !")
        print(train.shape)


    return oxuva_train, oxuva_test, train_size, val_size
