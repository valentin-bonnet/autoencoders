import tensorflow as tf
import glob

def oxuva_to_tfrecord(path):

    record_file = '/media/valentin/DATA1/Programmation/Datasets/images_dev/images/tfrecords/0001.tfrecords'
    images = glob.glob(path+"*.jpeg")
    tfre_options = tf.io.TFRecordOptions(compression_type="GZIP")
    i =1
    with tf.io.TFRecordWriter(record_file, options=tfre_options) as writer:
        for filename in images:
            if i % 100 == 0:
                print(i)
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string)
            writer.write(tf_example.SerializeToString())
            i = i + 1


def image_example(image_string):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

oxuva_to_tfrecord("/media/valentin/DATA1/Programmation/Datasets/images_dev/images/dev/vid0000/")