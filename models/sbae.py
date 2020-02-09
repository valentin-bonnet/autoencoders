import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions



class SBAE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=512, input_shape=64):
        super(SBAE, self).__init__()
        self.latent_dim = latent_dim

        ## ENCODER L2AB
        self.L2ab = tf.keras.Sequential()
        self.L2ab.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            self.L2ab.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.L2ab.add(tf.keras.layers.Flatten())
        self.L2ab.add(tf.keras.layers.Dense(latent_dim))

        ## ENCODER AB2L
        self.ab2L = tf.keras.Sequential()
        self.ab2L.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 2)))
        for l in layers:
            self.ab2L.add(
                tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.ab2L.add(tf.keras.layers.Flatten())
        self.ab2L.add(tf.keras.layers.Dense(latent_dim))


        layers.reverse()
        size_decoded_frame = int(input_shape/(2**len(layers)))
        size_decoded_layers = int(layers[0]/2)

        ## DECODER_L2AB

        self.L2ab.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        self.L2ab.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        ## DECODER_AB2L

        self.ab2L.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        self.ab2L.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        print("####")
        self.L2ab.summary()
        print("\n\n ####")
        self.ab2L.summary()


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
        L, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
        ab_logits = self.L2ab(L)
        L_logits = self.ab2L(ab)
        x_logit = tf.concat([L_logits, ab_logits], axis=-1)

        return x_logit

    def quantize(self, lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 101, 101)) - 1
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-87, 99, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 32 + lab_images[:, :, :, 2]
        return l_labels.reshape([-1, 32*32]), ab_labels.reshape([-1, 32*32])

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