import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sklearn.neighbors as nn



class AE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=512, input_shape=32, use_bn=False):
        super(AE, self).__init__()
        self.latent_dim = latent_dim

        ## ENCODER
        self.ae = tf.keras.Sequential()
        self.ae.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 3)))
        for l in layers:
            self.ae.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.ae.add(tf.keras.layers.BatchNormalization())
            self.ae.add(tf.keras.layers.ReLU())

        self.ae.add(tf.keras.layers.Flatten())
        self.ae.add(tf.keras.layers.Dense(latent_dim))



        layers.reverse()
        size_decoded_frame = int(input_shape/(2**len(layers)))
        size_decoded_layers = int(layers[0]/2)

        ## DECODER

        self.ae.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        self.ae.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.ae.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.ae.add(tf.keras.layers.BatchNormalization())
            self.ae.add(tf.keras.layers.ReLU())


        self.ae.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same'))
        if use_bn:
            self.ae.add(tf.keras.layers.BatchNormalization())
        self.ae.add(tf.keras.layers.ReLU())


        print("####")
        self.ae.summary()



        """self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
                tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
                tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim+latent_dim)

            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=4 * 4 * 128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 128)),
                tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
                tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
            ]
        )"""

    def reconstruct(self, x):
        return self.ae(x)


    def compute_loss(self, x):
        x_logits = self.reconstruct(x)
        loss = tf.reduce_sum(tf.square(x - x_logits))

        """
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)"""

        return loss

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compute_accuracy(self, x):
        x_logits = self.reconstruct(x)
        accuracy = tf.reduce_mean(tf.square(x_logits - x))

        return accuracy