import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sklearn.neighbors as nn

tfpl = tfp.layers
tfd = tfp.distributions



class SBAE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=512, input_shape=32, classification=False):
        super(SBAE, self).__init__()
        self.latent_dim = latent_dim
        self.cc = np.load('../utils/pts_in_hull.npy')
        self.nbrs = nn.NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(self.cc)

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

        if classification:
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=313, kernel_size=4, strides=tf.nn.softmax, activation=tf.nn.relu, padding='same'))
        else:
            self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))

        ## DECODER_AB2L

        self.ab2L.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        self.ab2L.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        for l in layers:
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'))

        if classification:
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=50, kernel_size=4, strides=2, activation=tf.nn.softmax, padding='same'))
        else:
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

    def quantize(self, lab_images):
        l = lab_images[:, :, 0]*50.0
        ab = (lab_images[:, :, :, 1:]*255.0)-128.0
        l = tf.searchsorted(l, np.linspace(0, 51, 51)) - 1
        bs, h, w, c = ab.shape()
        ab = tf.reshape(h*w, c)
        (dists, inds) = self.nbrs.kneighbors(ab)


        # Sigma = 5
        sigma=5
        wts = tf.exp(-dists ** 2 / (2 * sigma ** 2))
        wts = tf.reduce_mean(wts, axis=1)

        p_inds = np.arange(0, P, dtype='int')[:, np.newaxis]
        pts_enc_flt[p_inds, inds] = wts
        #pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

        """
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-110, 99, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 16 + lab_images[:, :, :, 2]
        print("L_LABELS: {0}".format(l_labels.shape))
        print("AB_LABLES: {0}".format(ab_labels.shape))
        return l_labels.reshape([-1, 16*16]), ab_labels.reshape([-1, 16*16])"""

    def dequantize(self, lab_images):
        print(lab_images.shape)
        # print(lab_images)
        l_vals = np.linspace(0, 100, 100)
        print(l_vals.shape)
        a_vals = np.linspace(-87, 99.1, 16)
        b_vals = np.linspace(-108, 95.1, 16)
        a = (lab_images[:, :, 1] / 16).astype(int)
        b = (lab_images[:, :, 1] % 16).astype(int)
        print(lab_images[:, :, 0])
        lab_images[:, :, 0] = l_vals[lab_images[:, :, 0]]
        lab_images[:, :, 1] = a_vals[a]
        concat = b_vals[b].reshape((16, 16, 1))
        print(lab_images.shape)
        print(concat.shape)
        result = np.concatenate((lab_images, concat), axis=2)
        print(result)
        return result

    def reconstruct(self, x, classification=False):
        L, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
        if classification:
            x_logits = 2
        else:
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

    def compute_loss(self, x, classification=False):
        if classification:
            x_logits = self.reconstruct(x, classification)
            loss = tf.reduce_sum(tf.square(x - x_logits))
        else:
            x_logits = self.reconstruct(x)
            loss = tf.reduce_sum(tf.square(x - x_logits))

        """
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)"""

        return loss

    def compute_apply_gradients(self, x, optimizer, is_Lab=False):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, is_Lab)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compute_accuracy(self, x):
        x_logits = self.reconstruct(x)

        accuracy = tf.reduce_mean(tf.square(x_logits - x))
        return accuracy