import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE

def imagenetresized64loader():
    data = tfds.load('imagenet_resized/64x64')
    train_data, test_data = data['train'], data['test']
    train_dataset = train_data.map(lambda x: x/255.0, num_parallel_calls=AUTOTUNE)
    test_dataset = test_data.map(lambda x: x/255.0, num_parallel_calls=AUTOTUNE)

    return train_dataset, test_dataset

def imagenetresized64loaderLab():
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    train_dataset = cv2.cvtColor(train_images / 255.0, cv2.COLOR_RGB2Lab)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset.astype('float32'))
    test_dataset = cv2.cvtColor(test_images / 255.0, cv2.COLOR_RGB2Lab)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset.astype('float32'))
    return train_dataset, test_dataset