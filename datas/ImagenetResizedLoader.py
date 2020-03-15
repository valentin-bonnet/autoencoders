import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _rgb2lab(image):
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2Lab)
    return image

def _preprocess_Lab(image):
    print(image['image'])
    image = tf.cast(image['image'], dtype=tf.float32)/255.0
    image = tf.py_function(func=_rgb2lab, inp=[image], Tout=tf.float32)
    image = image+[0, 128, 128]
    print(image.numpy())
    image = image.numpy()/[100.0, 255.0, 255.0]
    #image = _rgb2lab(image.numpy())
    return image

def imagenetresized64loader():
    data = tfds.load('imagenet_resized/64x64')
    train_data = data['train']
    test_data = data['validation']
    train_dataset = train_data.map(lambda x: x/255.0, num_parallel_calls=AUTOTUNE)
    test_dataset = test_data.map(lambda x: x/255.0, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset

def imagenetresized64loaderLab():
    data = tfds.load('imagenet_resized/64x64')
    train_dataset = data['train']
    print("####\n\ntrain done")
    test_dataset = data['validation']
    train_dataset = train_dataset.map(_preprocess_Lab, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(_preprocess_Lab, num_parallel_calls=AUTOTUNE)
    return train_dataset, test_dataset

