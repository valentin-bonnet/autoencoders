import tensorflow as tf
import oxuvaTFRecord
from skimage import color
import numpy as np
import glob

_imgs_per_folder = [4172, 2552, 1442, 10862, 3182, 1442, 2342, 4142, 3632, 1352, 1382, 1412, 8132, 1892, 1472, 1892, 1442, 4082, 2342, 9992, 1112, 3902, 3182, 1442, 37442, 4952, 1442, 4142, 2792, 1442, 3692, 6842, 13592, 2192, 8702, 1442, 12512, 5222, 1382, 1442, 10892, 1052, 9032, 12692, 9482, 4142, 992, 13082, 1322, 1262, 1052, 8642, 1982, 6392, 3242, 2342, 1382, 1892, 1052, 11792, 8912, 1892, 6842, 12212, 7741, 6842, 4982, 3182, 3692, 5972, 6392, 1442, 2732, 15812, 3242, 7292, 5372, 1412, 1232, 14492, 2702, 3152, 1532, 3182, 10892, 5492, 2702, 1742, 1442, 2672, 1292, 5492, 15842, 5882, 1382, 1382, 1232, 3242, 2582, 1442, 1382, 1442, 16592, 1382, 4082, 3242, 1382, 1382, 2342, 5312, 1262, 2342, 8192, 2552, 4142, 1352, 1292, 2282, 1832, 4592, 1442, 3902, 1382, 5852, 10082, 2792, 18092, 1442, 19052, 2342, 1442, 1172, 7322, 2342, 2192, 4082, 1382, 1442, 1382, 1382, 2342, 1442, 1382, 9032, 3062, 1382, 1442, 3182, 2372, 2432, 2732, 2342, 1352, 5492, 1412, 1442, 5432, 5042, 3632, 13532, 3122, 1892, 1442, 2792, 1442, 4142, 1382, 1292, 4142, 2282, 1442, 1442, 1322, 2252, 1352, 1892, 1082, 1382, 5042, 1952, 29342, 1802, 1442, 4442, 1442, 13982, 5492, 2732, 13142, 5012, 5942, 1442, 1442, 1382, 5942, 1382, 1442, 1502, 11282, 3242, 9992, 4982, 1442, 1742, 1442, 2342, 1412, 7172, 4592, 7742, 1442, 4142, 3182, 4142, 1442, 5432, 2132, 1232, 2792, 13112, 3182, 3032, 9542, 9542, 1442, 9032, 2762, 2282, 3902, 1442, 3122, 3692, 1442, 4082, 1472, 1532, 3632, 1322, 9482, 12602, 4142, 1352, 1472, 1442, 1442, 1532, 2792, 3242, 4622, 1292, 1442, 4742, 18541, 1352, 1382, 1382, 1382, 3242, 2282, 2342, 2612, 5942, 9902, 4112, 1622, 3692, 6392, 1442, 1442, 1772, 1442, 1382, 1442, 3242, 5042, 2342, 1172, 2852, 10952, 2282, 2282, 2282, 9032, 5492, 12182, 8132, 1442, 4142, 4142, 1442, 4982, 1442, 1442, 2792, 9032, 1292, 4532, 1442, 1202, 7292, 1502, 1322, 1292, 2342, 2792, 1352, 4142, 1382, 1382, 3332, 5942, 2762, 1862, 5432, 1892, 15392, 10832, 2222, 2762, 1442, 3242, 1352, 902, 1412, 3242, 1592, 1892, 11702, 4592, 1382, 1352, 1382, 2762, 1442, 2732, 1382, 1382]

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
    ds = ds.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(sequence_size*frames_delta)
    ds = ds.map(_preprocess_once, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def _get_size(size, f):
    print(f.numpy())
    files_number = int(f[-14:-10])
    files_size = _imgs_per_folder[files_number]
    true_size = files_size // (frames_delta*sequence_size)
    return true_size + size

def oxuva_loader_v2(path='/content/drive/My Drive/Colab Data/Datasets/oxuva_256/'):
    files = glob.glob(path+'*.tfrecords')
    ds_files = tf.data.Dataset.from_tensor_slices(files).shuffle(337, seed=1)
    i = 0
    true_seq_size = sequence_size*frames_delta
    np_size = np.array(_imgs_per_folder)
    nb_batch = np_size // true_seq_size
    all_size = np.sum(nb_batch)

    oxuva_val = ds_files.take(34)
    oxuva_train = ds_files.skip(34)

    val_size = oxuva_val.reduce(0, _get_size).numpy()

    oxuva_train = oxuva_train.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    oxuva_val = oxuva_val.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    train_size = all_size - val_size

    return oxuva_train, oxuva_val, train_size, val_size

def oxuva_loader(path='/content/drive/My Drive/Colab Data/Datasets/oxuva_256/', seq_size=8):
    datasets = oxuvaTFRecord.tfrecord_to_dataset(path)
    i = 0
    true_seq_size = seq_size*frames_delta
    np.random.seed(1)
    random_i = np.random.choice(len(_imgs_per_folder), len(_imgs_per_folder)//10, replace=False)
    np_size = np.array(_imgs_per_folder)
    nb_batch = np_size // true_seq_size
    all_size = np.sum(nb_batch)
    val_size = np.sum(nb_batch[random_i])

    oxuva_train = None
    oxuva_val = None

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
