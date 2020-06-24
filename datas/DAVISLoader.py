import tensorflow as tf
import glob

#frames_delta = 4
#frames_delta = 4
#sequence_size = 10
palette = tf.constant([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]])
new_palette = tf.reduce_sum(palette * [1, 10, 100], -1)

table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(new_palette, [0, 1, 2, 3, 4, 5, 6, 7, 8]), -1)

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


def _preprocess_once_256(example_proto):
    image_feature_description = {
        'image_jpeg': tf.io.FixedLenFeature([], tf.string),
        'annotation': tf.io.FixedLenFeature([], tf.string),

    }
    parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
    #jpeg= parsed_data['image_jpeg'][::frames_delta]

    jpeg= parsed_data['image_jpeg']
    #anno= parsed_data['annotation'][::frames_delta]
    anno= parsed_data['annotation']

    jpeg = tf.map_fn(tf.image.decode_jpeg, tf.reshape(jpeg, [-1]), dtype=tf.uint8)
    anno = tf.map_fn(tf.image.decode_png, tf.reshape(anno, [-1]), dtype=tf.uint8)
    anno = tf.cast(anno, tf.int32)
    anno = table.lookup(tf.reduce_sum(anno * [1, 10, 100], -1))
    anno_hot = tf.one_hot(anno, 9)
    anno_hot = tf.cast(anno_hot, tf.float32)
    jpeg = tf.cast(jpeg, tf.float32) / 255.0
    #anno = tf.cast(anno, tf.float32) / 255.0
    jpeg_lab = tf_rgb2lab(jpeg)
    #anno_lab = tf_rgb2lab(anno)
    jpeg_lab = tf.cast(jpeg_lab, tf.float32)
    #anno_lab = tf.cast(anno_lab, tf.float32)
    jpeg_lab = jpeg_lab + [0., 128.0, 128.0]
    jpeg_lab = (jpeg_lab / [50.0, 127.5, 127.5]) - 1.0
    #anno_lab = anno_lab + [0., 128.0, 128.0]
    #anno_lab = (anno_lab / [50.0, 127.5, 127.5]) - 1.0
    jpeg_lab = tf.reshape(jpeg_lab, [256, 256, 3])
    anno_hot = tf.reshape(anno_hot, [256, 256, 9])
    return jpeg_lab, anno_hot

def _preprocess_once(example_proto):
    image_feature_description = {
        'image_jpeg': tf.io.FixedLenFeature([], tf.string),
        'annotation': tf.io.FixedLenFeature([], tf.string),
        'h': tf.io.FixedLenFeature([], tf.int64),
        'w': tf.io.FixedLenFeature([], tf.int64)

    }
    parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
    #jpeg= parsed_data['image_jpeg'][::frames_delta]

    jpeg = parsed_data['image_jpeg']
    #anno= parsed_data['annotation'][::frames_delta]
    anno = parsed_data['annotation']
    h = parsed_data['h']
    w = parsed_data['w']

    jpeg = tf.map_fn(tf.image.decode_jpeg, tf.reshape(jpeg, [-1]), dtype=tf.uint8)
    anno = tf.map_fn(tf.image.decode_png, tf.reshape(anno, [-1]), dtype=tf.uint8)
    anno = tf.cast(anno, tf.int32)
    anno = table.lookup(tf.reduce_sum(anno * [1, 10, 100], -1))
    anno_hot = tf.one_hot(anno, 9)
    anno_hot = tf.cast(anno_hot, tf.float32)
    jpeg = tf.cast(jpeg, tf.float32) / 255.0
    #anno = tf.cast(anno, tf.float32) / 255.0
    jpeg_lab = tf_rgb2lab(jpeg)
    #anno_lab = tf_rgb2lab(anno)
    jpeg_lab = tf.cast(jpeg_lab, tf.float32)
    #anno_lab = tf.cast(anno_lab, tf.float32)
    jpeg_lab = jpeg_lab + [0., 128.0, 128.0]
    jpeg_lab = (jpeg_lab / [50.0, 127.5, 127.5]) - 1.0
    #anno_lab = anno_lab + [0., 128.0, 128.0]
    #anno_lab = (anno_lab / [50.0, 127.5, 127.5]) - 1.0
    jpeg_lab = tf.reshape(jpeg_lab, [h, w, 3])
    anno_hot = tf.reshape(anno_hot, [h, w, 9])
    return jpeg_lab, anno_hot




def _files_to_ds(f):
    ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
    #ds = ds.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # .batch(sequence_size*frames_delta, drop_remainder=True)
    #ds = ds.map(_preprocess_once, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(105)
    ds = ds.map(_preprocess_once, num_parallel_calls=tf.data.experimental.AUTOTUNE).window(35, 30)
    ds = ds.flat_map(lambda x: x.batch(35))
    return ds

def davis_loader(path='/content/drive/My Drive/Colab Data/Datasets/DAVIS_VAL_BIG/'):
    files = glob.glob(path+'*.tfrecords')
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    #davis_ds = ds_files.interleave(_files_to_ds, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    davis_ds = ds_files.flat_map(_files_to_ds)
    # davis_ds = ds_files.map(_files_to_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return davis_ds

