import tensorflow as tf
import oxuvaTFRecord
from skimage import color
import numpy as np
import glob

frames_delta = 4
sequence_size = 8

def _rgb2lab(image):
    return color.rgb2lab(image)

def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def tf_rgb2lab(image):
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(image)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                        ((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                        xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))

def _preprocess_one_ds(parsed_data):
    img = parsed_data['image_raw']
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32) / 255.0
    #img_lab = tf.py_function(func=_rgb2lab, inp=[img], Tout=tf.float32)
    img_lab = tf_rgb2lab(img)
    img_lab = tf.cast(img_lab, tf.float32)
    img_normalized = img_lab + [0., 128.0, 128.0]
    img_normalized = (img_normalized / [50.0, 127.5, 127.5]) - 1.0
    img_normalized = tf.reshape(img_normalized, [256, 256, 3])
    return img_normalized

def _preprocess_sequence_ds(parsed_batch):
    return parsed_batch[::frames_delta]

def _preprocess_once(parsed_data):
    img = parsed_data['image_raw'][::frames_delta]
    img = tf.map_fn(tf.image.decode_jpeg, img, dtype=tf.uint8)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.cast(img, tf.float32) / 255.0
    # img_lab = tf.py_function(func=_rgb2lab, inp=[img], Tout=tf.float32)
    img_lab = tf_rgb2lab(img)
    img_lab = tf.cast(img_lab, tf.float32)
    img_normalized = img_lab + [0., 128.0, 128.0]
    img_normalized = (img_normalized / [50.0, 127.5, 127.5]) - 1.0
    img_normalized = tf.reshape(img_normalized, [8, 256, 256, 3])
    return img_normalized


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  image_feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
  #print(parsed_data)
  #img = parsed_data['image_raw'].bytes_list.value[0]
  return parsed_data

def _files_to_ds(f):
    ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
    ds = ds.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(sequence_size*frames_delta, drop_remainder=True)
    ds = ds.map(_preprocess_once, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def _get_size(f, size):
    files_number = int(f.numpy()[-14:-10])
    files_size = _imgs_per_folder[files_number]
    true_size = files_size // (frames_delta*sequence_size)
    return true_size + size

def davis_loader(path='/content/drive/My Drive/Colab Data/Datasets/DAVIS/'):
    files = glob.glob(path+'*.tfrecords')
    ds_files = tf.data.Dataset.from_tensor_slices(files).shuffle(337, seed=1)
    true_seq_size = sequence_size*frames_delta


    oxuva_train = ds_files.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    oxuva_val = oxuva_val.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    train_size = all_size - val_size

    return oxuva_train, oxuva_val, train_size, val_size

def davis_loader(path='/content/drive/My Drive/Colab Data/Datasets/DAVIS/', seq_size=8):
    datasets = oxuvaTFRecord.tfrecord_to_dataset(path)
    i = 0
    true_seq_size = seq_size*frames_delta
    np.random.seed(1)
    davis_ds =

    for data in datasets:
        ds = data.batch(true_seq_size, drop_remainder=True)
        ds = ds.map(_preprocess_once)
        #ds = ds.map(_preprocess_sequence_ds)
        #ds = ds.map(_preprocess_one_ds)
        if i == 0:
            if i in random_i:
                oxuva_val = ds
            else:
                oxuva_train = ds
        else:
            if i in random_i:
                if oxuva_val is None:
                    oxuva_val = ds
                else:
                    oxuva_val = oxuva_val.concatenate(ds)
            else:
                if oxuva_train is None:
                    oxuva_train = ds
                else:
                    oxuva_train = oxuva_train.concatenate(ds)
        i = i +1


    train_size = all_size - val_size

    return oxuva_train, oxuva_val, train_size, val_size
