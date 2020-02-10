import tensorflow as tf
import numpy as np
import cv2


def _rgb2lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)


def cifarloader():
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    train_dataset = train_images / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset.astype('float32'))
    test_dataset = test_images / 255.0
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset.astype('float32'))
    return train_dataset, test_dataset

def cifarloaderLab():
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    train_images = np.float32(train_images) / 255.0
    test_images = np.float32(test_images) / 255.0
    train_dataset = np.array([_rgb2lab(image) for image in train_images])
    test_dataset = np.array([_rgb2lab(image) for image in test_images])
    train_dataset = train_dataset+[0, 128, 128]
    train_dataset = np.float32(train_dataset/[100.0, 255.0, 255.0])
    test_dataset = test_dataset+[0, 128, 128]
    test_dataset = np.float32(test_dataset/[100.0, 255.0, 255.0])
    #train_dataset = cv2.cvtColor(train_images, cv2.COLOR_RGB2Lab)
    #test_dataset = cv2.cvtColor(test_images, cv2.COLOR_RGB2Lab)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
    return train_dataset, test_dataset