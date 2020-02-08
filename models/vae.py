import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(64, 64, 3), name='Input1'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2000, activation="tanh"),
                tf.keras.layers.Dense(500, activation="tanh"),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim)),
                tf.keras.layers.Dense(500, activation='tanh'),
                tf.keras.layers.Dense(2000, activation="tanh"),
                # No activation
                tf.keras.layers.Dense(64 * 64 * 3),
                tf.keras.layers.Reshape((64, 64, 3)),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
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

    @tf.function
    def compute_loss(self, x, training=True):
      mean, logvar = self.encode(x, training=training)
      z = self.reparameterize(mean, logvar)
      x_logit = self.decode(z, training)

      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
      logpz = self.log_normal_pdf(z, 0., 0.)
      logqz_x = self.log_normal_pdf(z, mean, logvar)
      return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
      log2pi = tf.math.log(2. * np.pi)
      return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

    @tf.function
    def compute_apply_gradients(self, x, optimizer):
      with tf.GradientTape() as tape:
        loss = self.compute_loss(x)
      gradients = tape.gradient(loss, self.trainable_variables)
      optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      return loss

    @tf.function
    def compute_accuracy(self, x, training=True):
      mean, logvar = self.encode(x, training)
      z = self.reparameterize(mean, logvar)
      x_logit = self.decode(z, apply_sigmoid=True, training=training)
      accuracy = tf.reduce_mean(tf.square(x_logit - x))
      return accuracy