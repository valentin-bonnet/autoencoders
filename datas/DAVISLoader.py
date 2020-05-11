import tensorflow as tf
import glob

frames_delta = 4
sequence_size = 2


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


def _preprocess_once(parsed_data):
    jpeg= parsed_data['image_jpeg'][::frames_delta]
    anno= parsed_data['annotation'][::frames_delta]
    jpeg = tf.map_fn(tf.image.decode_jpeg, jpeg, dtype=tf.uint8)
    anno = tf.map_fn(tf.image.decode_png, anno, dtype=tf.uint8)
    jpeg = tf.cast(jpeg, tf.float32) / 255.0
    anno = tf.cast(anno, tf.float32) / 255.0
    # img_lab = tf.py_function(func=_rgb2lab, inp=[img], Tout=tf.float32)
    jpeg_lab = tf_rgb2lab(jpeg)
    anno_lab = tf_rgb2lab(anno)
    jpeg_lab = tf.cast(jpeg_lab, tf.float32)
    anno_lab = tf.cast(anno_lab, tf.float32)
    jpeg_lab = jpeg_lab + [0., 128.0, 128.0]
    jpeg_lab = (jpeg_lab / [50.0, 127.5, 127.5]) - 1.0
    jpeg_lab = tf.reshape(jpeg_lab, [sequence_size, 256, 256, 3])
    anno_lab = anno_lab + [0., 128.0, 128.0]
    anno_lab = (anno_lab / [50.0, 127.5, 127.5]) - 1.0
    anno_lab = tf.reshape(anno_lab, [sequence_size, 64, 64, 3])
    return jpeg_lab, anno_lab


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  image_feature_description = {
      'image_jpeg': tf.io.FixedLenFeature([], tf.string),
      'annotation': tf.io.FixedLenFeature([], tf.string)
  }
  parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
  return parsed_data

def _files_to_ds(f):
    ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
    ds = ds.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(sequence_size*frames_delta, drop_remainder=True)
    ds = ds.map(_preprocess_once, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def davis_loader(path='/content/drive/My Drive/Colab Data/Datasets/DAVIS/'):
    files = glob.glob(path+'*.tfrecords')
    ds_files = tf.data.Dataset.from_tensor_slices(files).shuffle(90, seed=1)
    davis_ds = ds_files.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return davis_ds
