import tensorflow as tf


## ResNet18
class ResNet(tf.keras.Model):
    def __init__(self, name='Resnet'):
        super(ResNet, self).__init__()
        self.model = tf.keras.models.Sequential()
        with tf.name_scope('stage0'):
            self.model.add(tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False))
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
        with tf.name_scope('stage1'):
            self.model.add(ResidualUnit(64, 1))
            self.model.add(ResidualUnit(64, 1))
        with tf.name_scope('stage2'):
            self.model.add(ResidualUnit(128, 2))
            self.model.add(ResidualUnit(128, 1))
        with tf.name_scope('stage3'):
            self.model.add(ResidualUnit(256, 1, skip_use=True))
            self.model.add(ResidualUnit(256, 1))
        with tf.name_scope('stage4'):
            self.model.add(ResidualUnit(256, 1))
            self.model.add(ResidualUnit(256, 1))

    def call(self, inputs):
        return self.model(inputs)



class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", skip_use=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]
        elif skip_use:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

