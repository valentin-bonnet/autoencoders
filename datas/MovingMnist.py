import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE



def movingMnist():
    train_dataset, test_dataset = tfds.load(name='moving_mnist', data_dir='/content/drive/My Drive/Colab Data/Datasets/', split=['test[10%:100%]', 'test[0%:10%]'])
    train_dataset = train_dataset.map(lambda x: tf.cast(x['image_sequence'], tf.float32) / 255.0, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(lambda x: tf.cast(x['image_sequence'], tf.float32) / 255.0, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset