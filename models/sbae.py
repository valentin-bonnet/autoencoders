import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sklearn.neighbors as nn

tfpl = tfp.layers
tfd = tfp.distributions



class SBAE(tf.keras.Model):
    def __init__(self, layers=[128, 256, 512], latent_dim=512, input_shape=32, use_bn=True, classification=True):
        super(SBAE, self).__init__()
        self.latent_dim = latent_dim
        self.inp_shape = input_shape
        pts = np.load('../utils/pts_in_hull.npy')
        self.nbrs = nn.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(pts)
        self.cc = tf.convert_to_tensor(pts)
        self.wgts = tf.convert_to_tensor(np.load('../utils/prior_probs.npy'))
        self.architecture = layers.copy()
        self.is_cl = classification
        self.description = "SBAE"
        #str_arch = '_'.join(str(x) for x in self.architecture)
        #str_bn = 'BN' if use_bn else ''
        #str_class = 'class' if classification else ''
        #self.description = '_'.join(filter(None, ['SBAE', str_arch, 'lat' + str(self.latent_dim), str_bn, str_class]))

        ## ENCODER L2AB
        self.L2ab = tf.keras.Sequential()
        self.L2ab.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            #self.L2ab.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, padding='same'))
            if l[2] > 0:
                self.L2ab.add(tf.keras.layers.Conv2D(filters=l[0], kernel_size=l[1], strides=l[2], padding='same'))
            else:
                self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=l[0], kernel_size=l[1], strides=-l[2], padding='same'))
            if use_bn:
                self.L2ab.add(tf.keras.layers.BatchNormalization())
            self.L2ab.add(tf.keras.layers.ReLU())
        self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=313, kernel_size=1, strides=1, padding='same'))



        #self.L2ab.add(tf.keras.layers.Flatten())
        #self.L2ab.add(tf.keras.layers.Dense(latent_dim))

        ##AB2L
        self.ab2L = tf.keras.Sequential()
        self.ab2L.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 2)))
        for l in layers:
            if l[2] > 0:
                self.ab2L.add(tf.keras.layers.Conv2D(filters=l[0], kernel_size=l[1], strides=l[2], padding='same'))
            else:
                self.ab2L.add(
                    tf.keras.layers.Conv2DTranspose(filters=l[0], kernel_size=l[1], strides=-l[2], padding='same'))
            if use_bn:
                self.ab2L.add(tf.keras.layers.BatchNormalization())
            self.ab2L.add(tf.keras.layers.ReLU())
        self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=50, kernel_size=1, strides=1, padding='same'))


        #self.ab2L.add(tf.keras.layers.Flatten())
        #self.ab2L.add(tf.keras.layers.Dense(latent_dim))

        #size_decoded_frame = int(input_shape/(2**len(layers)))
        #size_decoded_layers = int(layers[0]/2)

        ## DECODER_L2AB

        #self.L2ab.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        #self.L2ab.add(tf.keras.layers.Dense(mid_shape_L2ab[0]*mid_shape_L2ab[1]*mid_shape_L2ab[2]))
        #self.L2ab.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        #self.L2ab.add(tf.keras.layers.Reshape(target_shape=mid_shape_L2ab))
        """
        for l in layers:
            #self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, padding='same'))
            self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=5, strides=1, padding='same'))
            if use_bn:
                self.L2ab.add(tf.keras.layers.BatchNormalization())
            self.L2ab.add(tf.keras.layers.ReLU())

        if classification:
            self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=313, kernel_size=5, strides=1, padding='same'))
        else:
            #self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))
            self.L2ab.add(tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=5, strides=1, activation=tf.nn.relu, padding='same'))

        ## DECODER_AB2L

        #self.ab2L.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        #self.ab2L.add(tf.keras.layers.Dense(mid_shape_ab2L[0]*mid_shape_ab2L[1]*mid_shape_ab2L[2]))
        #self.ab2L.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        #self.ab2L.add(tf.keras.layers.Reshape(target_shape=mid_shape_ab2L))
        for l in layers:
            #self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, padding='same'))
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=5, strides=1, padding='same'))
            if use_bn:
                self.ab2L.add(tf.keras.layers.BatchNormalization())
            self.ab2L.add(tf.keras.layers.ReLU())

        if classification:
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=50, kernel_size=5, strides=1, padding='same'))
        else:
            #self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, activation=tf.nn.relu, padding='same'))
            self.ab2L.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=1, activation=tf.nn.relu, padding='same'))
        """
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
        l = lab_images[:, :, :, :1]*50.0
        ab = (lab_images[:, :, :, 1:]*255.0)-128.0
        e = np.linspace(1, 50, 50, dtype=np.float32)
        linspace = np.tile(e, [ab.shape[0], ab.shape[1], ab.shape[2], 1])
        l_inds = tf.searchsorted(linspace, l)
        l_inds = tf.squeeze(l_inds)
        bs, h, w, c = ab.shape
        ab = tf.reshape(ab, [bs*h*w, c])
        (dists, inds) = self.nbrs.kneighbors(ab)


        # Sigma = 5
        sigma=5
        wts = tf.exp(dists ** 2 / (2 * sigma ** 2))
        #wts = tf.reduce_mean(wts, axis=1)
        wts = tf.nn.softmax(wts)
        inds = tf.expand_dims(inds, -1)
        batch_ind = tf.expand_dims(tf.range(0, bs*h*w, 1, dtype=tf.int64), -1)
        batch_ind = tf.tile(batch_ind, tf.constant([1, 5], tf.int32))
        batch_ind = tf.expand_dims(batch_ind, -1)
        inds = tf.concat([batch_ind, inds], axis=-1)

        #wts = tf.expand_dims(wts, -2)


        #hot_mixed_l = tf.scatter_nd(indices=l_inds, updates=1, shape=[128, 32, 32, 50])
        hot_mixed_l = tf.one_hot(l_inds, 50)
        hot_mixed_ab = tf.scatter_nd(indices=inds, updates=wts, shape=[bs*h*w, 313])
        hot_mixed_ab = tf.reshape(hot_mixed_ab, [bs, h, w, 313])


        return hot_mixed_l, hot_mixed_ab




        #p_inds = np.arange(0, P, dtype='int')[:, np.newaxis]
        #pts_enc_flt[p_inds, inds] = wts
        #pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)

    """
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
        return result"""

    def dequantize(self, l_hot, ab_hot):
        l = tf.cast(tf.math.argmax(l_hot, axis=-1), dtype=tf.float32)/50.0
        l = tf.expand_dims(l, -1)
        ab_ind = tf.expand_dims(tf.math.argmax(ab_hot, axis=-1), -1)
        ab = tf.cast(tf.gather_nd(self.cc, ab_ind), dtype=tf.float32)
        ab = (ab+128.0)/255.0
        lab_img = tf.concat([l, ab], axis=-1)
        return lab_img


    def reconstruct(self, x):
        L, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
        if self.is_cl:
            ab_logits = self.L2ab(L)
            L_logits = self.ab2L(ab)
            x_logit = self.dequantize(L_logits, ab_logits)
        else:
            ab_logits = self.L2ab(L)
            L_logits = self.ab2L(ab)
            x_logit = tf.concat([L_logits, ab_logits], axis=-1)

        return x_logit
    """
    def quantize(self, lab_images):
        lab_images[:, :, :, 0] = np.digitize(lab_images[:, :, :, 0], np.linspace(0, 101, 101)) - 1
        lab_images[:, :, :, 1] = np.digitize(lab_images[:, :, :, 1], np.linspace(-87, 99, 17)) - 1
        lab_images[:, :, :, 2] = np.digitize(lab_images[:, :, :, 2], np.linspace(-108, 95, 17)) - 1
        l_labels = lab_images[:, :, :, 0]
        ab_labels = lab_images[:, :, :, 1] * 32 + lab_images[:, :, :, 2]
        return l_labels.reshape([-1, 32*32]), ab_labels.reshape([-1, 32*32])
    """
    def compute_loss(self, x):
        if self.is_cl:
            l, ab = tf.split(x, num_or_size_splits=[1, 2], axis=-1)
            l_hot, ab_hot = self.quantize(x)
            ab_ind = tf.expand_dims(tf.math.argmax(ab_hot, axis=-1), -1)
            priors = tf.reduce_sum(tf.cast(tf.gather_nd(self.wgts, ab_ind), dtype=tf.float32))
            l_logit = self.ab2L(ab)
            ab_logit = self.L2ab(l)

            #print(tf.nn.softmax(tf.reshape(ab_logit, [128*32*32, 313]))[0])
            #print(tf.reshape(ab_hot, [128*32*32, 313])[0])
            cross_entropy_l = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(l_hot, l_logit))
            cross_entropy_ab = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(ab_hot, ab_logit))
            #print("LOSS")
            #print(cross_entropy_l)
            #print(cross_entropy_ab)
            #print(priors)
            loss = tf.reduce_sum(cross_entropy_l + priors*cross_entropy_ab)
        else:
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

