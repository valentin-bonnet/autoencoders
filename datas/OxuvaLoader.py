import tensorflow as tf
import oxuvaTFRecord
from skimage import io, color
import numpy as np

def _rgb2lab(image):
    return color.rgb2lab(image)

def _preprocess_one_ds(parsed_data):
    img = parsed_data['image_raw']
    img = tf.image.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img_lab = tf.py_function(func=_rgb2lab, inp=[img], Tout=tf.float32)
    img_normalized = tf.cast(img_lab, tf.float32) + [0., 128.0, 128.0]
    img_normalized = (img_normalized / [50.0, 127.5, 127.5]) - 1.0
    return img_normalized

def oxuva_loader(path='/content/drive/My Drive/Colab Data/Datasets/oxuva_256/', seq_size=8):
    datasets = oxuvaTFRecord.tfrecord_to_dataset(path)
    i = 1
    size = 0
    for data in datasets:

        ds = data.map(_preprocess_one_ds)
        ds = ds.batch(seq_size, drop_remainder=True)
        size_ds = len(list(ds))
        size = size + size_ds
        print(i, "\n\tsize dataset: ", size_ds, "\n\tsize tot: ", size)
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