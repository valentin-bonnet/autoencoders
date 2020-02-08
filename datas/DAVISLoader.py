from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

#catnames = []

def test_datapathloader(filepath):
    data_dir = _datapathloader(filepath)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    return image_count

def _datapathloader(filepath, txt_file):
    catname_txt = filepath + '/ImageSets/2017/'+txt_file+'.txt'

    global catnames
    catnames = open(catname_txt).readlines()

    jpeg_all = []

    for catname in catnames:

        jpeg_path = os.path.join(filepath, 'JPEGImages/480p/' + catname.strip())
        cat_jpegs = [os.path.join(jpeg_path, file) for file in sorted(os.listdir(jpeg_path))]
        jpeg_all.extend(cat_jpegs)

    return jpeg_all

def _decode_img(file_path):
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    #img = tf.image.rgb_to_grayscale(img)
    # resize the image to the desired size.
    img = tf.image.resize(img, [32, 32])
    return img



def get_dataset(filepath, txt_file):
    data_dir = _datapathloader(filepath, txt_file)
    list_ds = tf.data.Dataset.list_files(data_dir)
    img_ds = list_ds.map(_decode_img, num_parallel_calls=AUTOTUNE)
    return img_ds


