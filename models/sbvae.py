import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions



class SBVAE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=512, input_shape=32):
        super(SBVAE, self).__init__()
        self.latent_dim = latent_dim

        ## ENCODER L2AB
        self.inference_net_L2ab = tf.keras.Sequential()
        self.inference_net_L2ab.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            self.inference_net_L2ab.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.inference_net_L2ab.add(tf.keras.layers.Flatten())
        self.inference_net_L2ab.add(tf.keras.layers.Dense(latent_dim+latent_dim))

        ## ENCODER L2AB
        self.inference_net_ab2L = tf.keras.Sequential()
        self.inference_net_ab2L.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 2)))
        for l in layers:
            self.inference_net_ab2L.add(
                tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.inference_net_ab2L.add(tf.keras.layers.Flatten())
        self.inference_net_ab2L.add(tf.keras.layers.Dense(latent_dim + latent_dim))


        layers.reverse()
        size_decoded_frame = int(input_shape/(2**len(layers)))
        size_decoded_layers = int(layers[0]/2)

        ## DECODER_L2AB

        self.generative_net_L2ab = tf.keras.Sequential()
        self.generative_net_L2ab.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        self.generative_net_L2ab.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        self.generative_net_L2ab.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.generative_net_L2ab.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.generative_net_L2ab.add(tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        ## DECODER_AB2L

        self.generative_net_ab2L = tf.keras.Sequential()
        self.generative_net_ab2L.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        self.generative_net_ab2L.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        self.generative_net_ab2L.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.generative_net_ab2L.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        self.generative_net_ab2L.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        print("####")
        self.inference_net_ab2L.summary()
        self.inference_net_L2ab.summary()
        print("\n\n ####")
        self.generative_net_ab2L.summary()
        self.generative_net_L2ab.summary()


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

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode_L2ab(self, x):
        mean, logvar = tf.split(self.inference_net_L2ab(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def encode_ab2L(self, x):
        mean, logvar = tf.split(self.inference_net_ab2L(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode_ab2L(self, z, apply_sigmoid=False):
        logits = self.generative_net_ab2L(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def decode_L2ab(self, z, apply_sigmoid=False):
        logits = self.generative_net_L2ab(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def _gaussian_log_likelihood(self, targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.log(std)
        return se


    def quantize(self, lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 101, 101)) - 1
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-87, 99, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 32 + lab_images[:, :, :, 2]
        return l_labels.reshape([-1, 32*32]), ab_labels.reshape([-1, 32*32])

    def compute_loss(self, x):
        L, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
        mean_L, logvar_L = self.encode_L2ab(L)
        mean_ab, logvar_ab = self.encode_ab2L(ab)
        z_ab = self.reparameterize(mean_ab, logvar_ab)
        z_L = self.reparameterize(mean_L, logvar_L)
        ab_logit = self.decode_L2ab(z_L, apply_sigmoid=False)
        L_logit = self.decode_ab2L(z_ab, apply_sigmoid=False)

        x_logit = tf.concat([L_logit, ab_logit], axis=-1)

        reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
          tf.keras.layers.Flatten()(x_logit), scale_identity_multiplier=0.05).log_prob(tf.keras.layers.Flatten()(x)))

        kl_divergence = tf.reduce_sum(tf.keras.metrics.kullback_leibler_divergence(x, x_logit), axis=[1, 2])

        #cross_ent = self._gaussian_log_likelihood(x_logit, mean, logvar)
        """
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)"""
        return tf.reduce_mean(reconstruction_term + kl_divergence)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compute_accuracy(self, x):
        L, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
        mean_L, logvar_L = self.encode_L2ab(L)
        mean_ab, logvar_ab = self.encode_ab2L(ab)
        z_ab = self.reparameterize(mean_ab, logvar_ab)
        z_L = self.reparameterize(mean_L, logvar_L)
        ab_logit = self.decode_L2ab(z_L, apply_sigmoid=False)
        L_logit = self.decode_ab2L(z_ab, apply_sigmoid=False)

        x_logit = tf.concat([L_logit, ab_logit], axis=-1)

        accuracy = tf.reduce_mean(tf.square(x_logit - x))
        return accuracy