import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class RKN(tf.keras.Model):
    def __init__(self, layers=[32, 32, 32], latent_dim=256, input_shape=64, sequence_length=20, M=32, use_bn=False):
        super(RKN, self).__init__()
        self.model_type = 'RKN'
        self.batch_size = 256
        self.architecture = layers.copy()
        self.latent_dim = latent_dim
        self.use_bn = use_bn
        self.seq_size = sequence_length
        self.im_shape = input_shape
        self.B = 2
        self.K = 4
        self.M = M # Observation space (Latent space)
        self.N = 2*M # Z space

        str_arch = '_'.join(str(x) for x in self.architecture)
        str_bn = 'BN' if use_bn else ''
        self.description = '_'.join(filter(None, ['KVAE', str_arch, 'lat' + str(self.M), 'seq' + str(self.seq_size), str_bn]))

        init_z0 = tf.zeros((self.batch_size, self.N))
        init_std_u = tf.ones((self.batch_size, self.M)) * 10.0  # z*z
        init_std_l = tf.ones((self.batch_size, self.M)) * 10.0  # z*z
        init_std_s = tf.zeros((self.batch_size, self.M))
        init_std_trans_u = tf.ones((self.batch_size, self.M)) * 1.1
        init_std_trans_l = tf.ones((self.batch_size, self.M)) * 1.1

        B11_init = tf.eye(self.M, batch_shape=[self.K])
        B12_init = tf.eye(self.M, batch_shape=[self.K]) * 0.2
        B21_init = tf.eye(self.M, batch_shape=[self.K]) * -0.2
        B22_init = tf.eye(self.M, batch_shape=[self.K])

        self.z0 = tf.Variable(initial_value=init_z0, trainable=False)
        self.std_u = tf.Variable(initial_value=init_std_u, trainable=False)
        self.std_l = tf.Variable(initial_value=init_std_l, trainable=False)
        self.std_s = tf.Variable(initial_value=init_std_s, trainable=False)
        self.std_trans_u = tf.exp(tf.Variable(initial_value=init_std_trans_u, trainable=True))
        self.std_trans_l = tf.exp(tf.Variable(initial_value=init_std_trans_l, trainable=True))
        self.B11 = tf.linalg.band_part(tf.Variable(initial_value=B11_init), self.B, self.B)
        self.B12 = tf.linalg.band_part(tf.Variable(initial_value=B12_init), self.B, self.B)
        self.B21 = tf.linalg.band_part(tf.Variable(initial_value=B21_init), self.B, self.B)
        self.B22 = tf.linalg.band_part(tf.Variable(initial_value=B22_init), self.B, self.B)


        self.lgssm_parameters_inference = tf.keras.Sequential()
        self.lgssm_parameters_inference.add(tf.keras.layers.Input(shape=(None, self.N), batch_size=self.batch_size))
        self.lgssm_parameters_inference.add(tf.keras.layers.LSTM(32, stateful=True))
        self.lgssm_parameters_inference.add(tf.keras.layers.Flatten())
        self.lgssm_parameters_inference.add(tf.keras.layers.Dense(self.K, activation=tf.nn.softmax))

        ## ENCODER
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            self.inference_net.add(tf.keras.layers.Conv2D(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.inference_net.add(tf.keras.layers.BatchNormalization())
            self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(2*self.M))

        ## DECODER

        layers.reverse()
        size_decoded_frame = int(input_shape//(2**len(layers)))
        #size_decoded_frame = self.im_shape
        size_decoded_layers = int(layers[0]//2)

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(self.N,)))
        self.generative_net.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        self.generative_net.add(
            tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))

        for l in layers:
            self.generative_net.add(
                tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=4, strides=2, padding='same'))
            if use_bn:
                self.generative_net.add(tf.keras.layers.BatchNormalization())
            self.generative_net.add(tf.keras.layers.ReLU())

        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='same'))

        print("####")
        self.inference_net.summary()
        print("\n\n####")
        self.generative_net.summary()
        print("\n\n####")
        self.lgssm_parameters_inference.summary()

    def pred(self, z_post, std_u, std_l, std_s):
        alpha = self.lgssm_parameters_inference(tf.expand_dims(z_post, 1)) # (bs, K)
        B11 = tf.reshape(alpha @ tf.reshape(self.B11, [-1, self.M*self.M]), [-1, self.M, self.M]) # (bs, M, M)
        B12 = tf.reshape(alpha @ tf.reshape(self.B12, [-1, self.M*self.M]), [-1, self.M, self.M]) # (bs, M, M)
        B21 = tf.reshape(alpha @ tf.reshape(self.B21, [-1, self.M*self.M]), [-1, self.M, self.M]) # (bs, M, M)
        B22 = tf.reshape(alpha @ tf.reshape(self.B22, [-1, self.M*self.M]), [-1, self.M, self.M]) # (bs, M, M)
        A_pred = tf.concat([tf.concat([B11, B12], -1),
                            tf.concat([B21, B22], -1)], -2)
        z_prior = tf.squeeze(A_pred @ tf.expand_dims(z_post, -1))
        std_u_prior = tf.reduce_sum(tf.square(B11), -1)* std_u + 2*tf.reduce_sum(B11*B12, -1)* std_s + tf.reduce_sum(tf.square(B12), -1)*std_l + self.std_trans_u
        std_l_prior = tf.reduce_sum(tf.square(B21), -1)* std_u + 2*tf.reduce_sum(B22*B21, -1)* std_s + tf.reduce_sum(tf.square(B22), -1)*std_l + self.std_trans_l
        std_s_prior = tf.reduce_sum(B21*B11, -1)* std_u + tf.reduce_sum(B22*B11, -1)* std_s + tf.reduce_sum(B21*B12, -1)* std_s +tf.reduce_sum(B22*B12, -1)*std_l

        return z_prior, std_u_prior, std_l_prior, std_s_prior

    def update(self, z_prior, std_u_prior, std_l_prior, std_s_prior, a_mean, a_std):
        q_u = std_u_prior / (std_u_prior + a_std)
        q_l = std_s_prior / (std_u_prior + a_std)
        residual = a_mean - z_prior[:, :self.M]

        z_post = z_prior + tf.concat([q_u*residual, q_l*residual], -1)

        std_u_post = (1 - q_u) * std_u_prior
        std_l_post = std_l_prior - (q_u * std_s_prior)
        std_s_post = (1 - q_u) * std_s_prior

        return z_post, std_u_post, std_l_post, std_s_post

    def compute_loss(self, images):
        z_prev = self.z0
        std_u_prev = self.std_u
        std_l_prev = self.std_l
        std_s_prev = self.std_s
        loss = 0
        self.lgssm_parameters_inference.reset_states()
        for i in tf.range(self.seq_size):
            z_prior, std_u_prior, std_l_prior, std_s_prior = self.pred(z_prev, std_u_prev, std_l_prev, std_s_prev)
            mu_a, std_a = self.encode(images[:, i])
            z_post, std_u_post, std_l_post, std_s_post = self.update(z_prior, std_u_prior, std_l_prior, std_s_prior, mu_a, std_a)

            z_prev = z_post
            std_u_prev = std_u_post
            std_l_prev = std_l_post
            std_s_prev = std_s_post

            im_logit = self.decode(z_post, True)
            loss -= tf.reduce_sum(self.log_bernoulli(images[:, i], im_logit, eps=1e-6))

        return loss

    def compute_accuracy(self, images):
        z_prev = self.z0
        std_u_prev = self.std_u
        std_l_prev = self.std_l
        std_s_prev = self.std_s
        acc = 0
        self.lgssm_parameters_inference.reset_states()
        for i in tf.range(self.seq_size):
            z_prior, std_u_prior, std_l_prior, std_s_prior = self.pred(z_prev, std_u_prev, std_l_prev, std_s_prev)
            mu_a, std_a = self.encode(images[:, i])
            z_post, std_u_post, std_l_post, std_s_post = self.update(z_prior, std_u_prior, std_l_prior, std_s_prior,
                                                                     mu_a, std_a)

            z_prev = z_post
            std_u_prev = std_u_post
            std_l_prev = std_l_post
            std_s_prev = std_s_post

            im_logit = self.decode(z_post, True)
            acc += tf.reduce_sum(tf.square(images[:, i] - im_logit))

        return acc

    def reconstruct(self, images):
        z_prev = self.z0
        std_u_prev = self.std_u
        std_l_prev = self.std_l
        std_s_prev = self.std_s
        self.lgssm_parameters_inference.reset_states()
        output = []
        for i in tf.range(self.seq_size):
            z_prior, std_u_prior, std_l_prior, std_s_prior = self.pred(z_prev, std_u_prev, std_l_prev, std_s_prev)
            mu_a, std_a = self.encode(images[:, i])
            z_post, std_u_post, std_l_post, std_s_post = self.update(z_prior, std_u_prior, std_l_prior, std_s_prior,
                                                                     mu_a, std_a)

            z_prev = z_post
            std_u_prev = std_u_post
            std_l_prev = std_l_post
            std_s_prev = std_s_post

            im_temp = self.decode(z_post, True)
            output.append(im_temp)
        images = tf.stack(output, axis=1)
        return images

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def log_bernoulli(self, x, p, eps=0.0):
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return x * tf.math.log(p) + (1 - x) * tf.math.log(1 - p)


    def encode(self, a):
        a_inf = self.inference_net(a)
        # tf.print("x_inf : ", x_inf[0])
        mean, std = tf.split(a_inf, num_or_size_splits=[self.M, self.M], axis=1)
        std = tf.nn.sigmoid(std) * 0.03
        return mean, std

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape, dtype=tf.float32)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits






