import tensorflow as tf
import oxuvaTFRecord
import cv2
import numpy as np

def _rgb2lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

def _preprocess_one_ds(img):
    img_lab = _rgb2lab(img)
    img_normalized = tf.cast(img_lab, tf.float32) + [0., 128.0, 128.0]
    img_normalized = (img_normalized / [50.0, 127.5, 127.5]) - 1.0
    return img_normalized

def oxuva_loader(path='/content/drive/My Drive/Colab Data/Datasets/oxuva_256/', seq_size=8):
    datasets = oxuvaTFRecord.tfrecord_to_dataset(path)
    i = 1
    size = 0
    for data in datasets:
        print(i)
        ds = data.map(_preprocess_one_ds)
        ds = ds.batch(seq_size, drop_remainder=True)
        size = size + len(list(ds))
        if i == 1:
            oxuva = ds
        else:
            oxuva.concatenate(ds)
        i = i +1
    print("#### \n\nSIZE:", size, "####")

    oxuva = oxuva.shuffle(100000, seed=1)
    oxuva_test = oxuva.take(10000)
    oxuva_train = oxuva.skip(10000)

    return oxuva_train, oxuva_test