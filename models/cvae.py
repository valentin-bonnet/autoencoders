import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions



class CVAE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=1024, input_shape=32, use_bn=False):
        super(CVAE, self).__init__()
        self.model_type = 'VAE'
        self.architecture = layers.copy()
        self.latent_dim = latent_dim
        self.use_bn = use_bn

        str_arch = '_'.join(str(x) for x in self.architecture)
        str_bn = 'BN' if use_bn else ''
        self.description = '_'.join(filter(None, ['VAE', str_arch, 'lat'+str(self.latent_dim), str_bn]))

        ## ENCODER
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 3)))
        for l in layers:
            self.inference_net.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.inference_net.add(tf.keras.layers.BatchNormalization())
            self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(latent_dim+latent_dim))

        ## DECODER

        layers.reverse()
        size_decoded_frame = int(input_shape//(2**len(layers)))
        size_decoded_layers = int(layers[0]//2)

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        self.generative_net.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        self.generative_net.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))


        for l in layers:
            self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.generative_net.add(tf.keras.layers.BatchNormalization())
            self.generative_net.add(tf.keras.layers.ReLU())

        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        print("####")
        self.inference_net.summary()
        print("\n\n ####")
        self.generative_net.summary()


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
            eps = tf.random.normal(shape=(10000, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, training=True):
        mean, logvar = tf.split(self.inference_net(x, training), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def _gaussian_log_likelihood(self, targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.log(std)
        return se

    def reconstruct(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=False)
        return x_logit

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=False)


        #reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
        #  tf.keras.layers.Flatten()(x_logit), scale_identity_multiplier=0.05).log_prob(tf.keras.layers.Flatten()(x)))

        #kl_divergence = tf.reduce_sum(tf.keras.metrics.kullback_leibler_divergence(x, x_logit), axis=[1, 2])

        #cross_ent = self._gaussian_log_likelihood(x_logit, mean, logvar)
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

        reconstr = tf.reduce_mean(tf.square(x - x_logit))
        kl = self._kl_diagnormal_stdnormal(mean, logvar)
        return tf.reduce_mean(reconstr + kl)

        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        #logpx_z = tf.reduce_mean(tf.square(x - x_logit))
        #logpz = self.log_normal_pdf(z, 0., 0.)
        #logqz_x = self.log_normal_pdf(z, mean, logvar)
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x)
        #return tf.reduce_mean(reconstruction_term + kl_divergence)
        #return tf.reduce_sum(tf.square(x - x_logit))

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def _kl_diagnormal_stdnormal(self, mu, log_var):
        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compute_accuracy(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=True)
        accuracy = tf.reduce_mean(tf.square(x_logit - x))
        return accuracy

class View_VAE():
    def init(self, vae):
        self.vae = vae