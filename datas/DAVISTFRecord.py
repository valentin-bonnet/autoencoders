import tensorflow as tf
import glob
from PIL import Image
import os

def DAVIS_to_tfrecord(path_davis_jpeg, path_davis_anno, path_tfre):
    print("#####")
    subfolders_jpeg = [f.path for f in os.scandir(path_davis_jpeg) if f.is_dir()]
    subfolders_anno = [f.path for f in os.scandir(path_davis_anno) if f.is_dir()]
    subfolders = set(zip(subfolders_jpeg, subfolders_anno))
    print(subfolders)
    #record_file = '/media/valentin/DATA1/Programmation/Datasets/images_dev/images/tfrecords/0001_jpeg.tfrecords'
    for sub in subfolders:
        sub_anno = sub[0]
        sub_jpeg = sub[1]
        images_jpeg = glob.glob(sub_jpeg+"/*.jpeg")
        images_anno = glob.glob(sub_anno+"/*.jpeg")
        tfre_options = tf.io.TFRecordOptions(compression_type="GZIP")
        folder_name = os.path.basename(os.path.normpath(sub_anno+'/'))
        record_file = path_tfre+folder_name+'.tfrecords'
        size = len(images_jpeg)
        i= 1
        j = 10
        k = size//10
        with tf.io.TFRecordWriter(record_file, options=tfre_options) as writer:
            for idx, _ in enumerate(images_anno):
                if i == k:
                    print("\t", j, "%")
                    j = j+10
                    k = k + size//10
                anno_string = open(images_anno[i], 'rb').read()
                anno_tf = tf.io.decode_jpeg(anno_string)
                resized_anno_tf = tf.cast(tf.image.resize(anno_tf, (64, 64)), tf.uint8)  # RESIZE
                anno_resized_byte = tf.io.encode_png(resized_anno_tf)
                jpeg_string = open(images_jpeg[i], 'rb').read()
                jpeg_tf = tf.io.decode_jpeg(jpeg_string)
                resized_jpeg_tf = tf.cast(tf.image.resize(jpeg_tf, (256, 256)), tf.uint8)  # RESIZE
                jpeg_resized_byte = tf.io.encode_jpeg(resized_jpeg_tf)
                tf_example = image_example(anno_resized_byte, jpeg_resized_byte)
                writer.write(tf_example.SerializeToString())
                i = i + 1

def image_example(anno_string, jpeg_string):

    feature = {
        'image_jpeg': _bytes_feature(jpeg_string),
        'annotation': _bytes_feature(anno_string)
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


path_davis_jpeg = '/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p/'
path_davis_anno = '/media/valentin/DATA1/Programmation/Datasets/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/'
path_tfre = '/media/valentin/DATA1/Programmation/Datasets/DAVIS_TFRecords/'

DAVIS_to_tfrecord(path_davis_jpeg, path_davis_anno, path_tfre)