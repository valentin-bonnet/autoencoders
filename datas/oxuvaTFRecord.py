import tensorflow as tf
import glob
from PIL import Image
import os

def oxuva_to_tfrecord(path_oxuva, path_tfre):
    print("#####")
    subfolders = [f.path for f in os.scandir(path_oxuva) if f.is_dir()]
    #record_file = '/media/valentin/DATA1/Programmation/Datasets/images_dev/images/tfrecords/0001_jpeg.tfrecords'
    for sub in subfolders:
        print(sub[-7:])
        images = glob.glob(sub+"/*.jpeg")
        tfre_options = tf.io.TFRecordOptions(compression_type="GZIP")
        record_file = path_tfre+sub[-4:]+'.tfrecords'
        size = len(images)
        i= 1
        j = 10
        k = size//10
        with tf.io.TFRecordWriter(record_file, options=tfre_options) as writer:
            for filename in images:
                if i == k:
                    print("\t", j, "%")
                    j = j+10
                    k = k + size//10
                image_string = open(filename, 'rb').read()
                image_tf = tf.io.decode_jpeg(image_string)
                resized_image_tf = tf.cast(tf.image.resize(image_tf, (256, 256)), tf.uint8)  # RESIZE
                image_resized_byte = tf.io.encode_jpeg(resized_image_tf)
                tf_example = image_example(image_resized_byte)
                writer.write(tf_example.SerializeToString())
                i = i + 1

def _tfrecord_to_oxuva_dataset(path):
    raw_image_dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset.take(1):

        h = image_features['height']
        w = image_features['width']
        image_raw = tf.io.decode_jpeg(image_features['image_raw'])

    # Create a dictionary describing the features.

def tfrecord_to_dataset(path):
    files = glob.glob(path+'*.tfrecords')
    datasets = []
    for f in files:
        ds = tf.data.TFRecordDataset(f, compression_type="GZIP")
        ds = ds.map(_parse_image_function)
        datasets.append(ds)
    return datasets

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  image_feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  parsed_data = tf.io.parse_single_example(example_proto, image_feature_description)
  print(parsed_data)
  img = parsed_data['image_raw'].bytes_list.value[0]
  return img



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

#path_oxuva = "/media/valentin/DATA1/Programmation/Datasets/images_all/"
#path_tfre = "/media/valentin/DATA1/Programmation/Datasets/images_all_tfrecords/"
#oxuva_to_tfrecord(path_oxuva, path_tfre)
#tfrecord_to_oxuva('/media/valentin/DATA1/Programmation/Datasets/images_dev/images/tfrecords/0001_jpeg.tfrecords')