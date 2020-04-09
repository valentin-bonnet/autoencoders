import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions



class KVAE(tf.keras.Model):
    def __init__(self, layers=[64, 128, 512], latent_dim=1024, input_shape=64, sequence_length=20, dim_a=20, dim_z=4, dim_u=10, std=0.05, use_bn=False):
        super(KVAE, self).__init__()
        self.model_type = 'KVAE'
        self.batch_size = 8
        self.architecture = layers.copy()
        self.latent_dim = latent_dim
        self.std = std
        self.use_bn = use_bn
        self.seq_size = sequence_length
        self.im_shape = input_shape
        self.dim_a = dim_a
        self.dim_z = dim_z
        #self.dim_u = dim_u
        self.K = 3
        self.latent_dim = self.dim_a + (self.dim_a * (self.dim_a + 1) // 2)

        self.A = tf.eye(self.dim_z, batch_shape=[self.K], dtype=tf.float64)
        self.C = tf.random.normal(shape=[self.K, self.dim_a, self.dim_z], dtype=tf.float64)*0.05
        self.Q = tf.constant(tf.eye(self.dim_z, dtype=tf.float64) * 0.08)  # z*z
        self.R = tf.constant(tf.eye(self.dim_a, dtype=tf.float64) * 0.03)

        str_arch = '_'.join(str(x) for x in self.architecture)
        str_bn = 'BN' if use_bn else ''
        self.description = '_'.join(filter(None, ['KVAE', str_arch, 'lat'+str(self.latent_dim), 'seq'+str(self.seq_size), str_bn]))

        self.lgssm_parameters_inference = tf.keras.Sequential()

        self.lgssm_parameters_inference.add(tf.keras.layers.Input(shape=(None, self.dim_a), batch_size=self.batch_size, dtype='float64'))
        self.lgssm_parameters_inference.add(tf.keras.layers.LSTM(50, stateful=True, dtype='float64'))
        self.lgssm_parameters_inference.add(tf.keras.layers.Flatten())
        #self.lgssm_parameters_inference.add(tf.keras.layers.Dense(self.dim_z ** 2 + self.dim_z * self.dim_u + self.dim_a * self.dim_z))
        self.lgssm_parameters_inference.add(tf.keras.layers.Dense(self.K, activation=tf.nn.softmax, dtype='float64'))

        ## ENCODER
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            self.inference_net.add(tf.keras.layers.Conv2D(filters=l, kernel_size=5, strides=1, padding='same'))
            if use_bn:
                self.inference_net.add(tf.keras.layers.BatchNormalization())
            self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(self.dim_a*2))

        ## DECODER

        layers.reverse()
        size_decoded_frame = int(input_shape//(2**len(layers)))
        size_decoded_layers = int(layers[0]//2)

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(self.dim_a,)))
        self.generative_net.add(tf.keras.layers.Dense(size_decoded_frame*size_decoded_frame*size_decoded_layers))
        self.generative_net.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))


        for l in layers:
            self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=5, strides=1, padding='same'))
            if use_bn:
                self.generative_net.add(tf.keras.layers.BatchNormalization())
            self.generative_net.add(tf.keras.layers.ReLU())

        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='same'))

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

    def prediction(self, z_prev, std_prev, A):
        z_hat = tf.squeeze(tf.matmul(A, tf.expand_dims(z_prev, -1)))
        std_hat = tf.matmul(A, std_prev)
        std_hat = tf.matmul(std_hat, tf.transpose(A, perm=[0, 2, 1])) + self.Q
        return z_hat, std_hat

    def update(self, a, z_hat, std_hat, C):
        residual = tf.squeeze(a) - tf.squeeze(tf.matmul(C, tf.expand_dims(z_hat, -1)))
        residual_std = tf.matmul(C, std_hat)
        residual_std = tf.matmul(residual_std, tf.transpose(C, perm=[0, 2, 1])) + self.R
        K = tf.matmul(std_hat, tf.transpose(C, perm=[0, 2, 1]))
        K = tf.matmul(K, tf.linalg.inv(residual_std))
        post_z = z_hat + tf.squeeze(tf.matmul(K, tf.expand_dims(residual, -1)))
        #post_std = (tf.eye(self.dim_z, dtype=tf.float32) - tf.matmul(K, C))
        #post_std = tf.matmul(post_std, std_hat)
        IKC = (tf.eye(self.dim_z, dtype=tf.float64) - tf.matmul(K, C))
        post_std = IKC @ std_hat @ tf.transpose(IKC, [0, 2, 1]) + K @ self.R @ tf.transpose(K, [0, 2, 1])
        return post_z, post_std

    def smooth(self, images, z0, std0, a0):

        z_hat_arr = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        std_hat_arr = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        post_z_arr = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        post_std_arr = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        a_arr = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        std_min = tf.eye(self.dim_z, self.dim_z, [self.batch_size], dtype=tf.float64)*1e-4
        z_prev = z0
        std_prev = std0
        a_prev = a0

        A = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        C = tf.TensorArray(tf.float64, size=self.seq_size, clear_after_read=False)
        # for i, img in enumerate(images):
        self.lgssm_parameters_inference.reset_states()
        for i in tf.range(self.seq_size):
            alpha = self.lgssm_parameters_inference(a_prev)
            #Ai, Ci = tf.split(AC, num_or_size_splits=[self.dim_z ** 2, self.dim_a * self.dim_z], axis=-1)
            Ai = tf.reshape(tf.matmul(alpha, tf.reshape(self.A, [-1, self.dim_z*self.dim_z])), [-1, self.dim_z, self.dim_z])
            Ci = tf.reshape(tf.matmul(alpha, tf.reshape(self.C, [-1, self.dim_a * self.dim_z])), [-1, self.dim_a, self.dim_z])

            A = A.write(i, Ai)
            C = C.write(i, Ci)

            mu_a, std_a = self.encode(images[:, i])
            a_prev = self.reparameterize(mu_a, tf.math.log(std_a))
            a_arr = a_arr.write(i, a_prev)
            a_prev = tf.expand_dims(a_prev, 1)


            z_hat, std_hat = self.prediction(z_prev, std_prev, A.read(i))
            z_hat_arr = z_hat_arr.write(i, z_hat)
            std_hat_arr = std_hat_arr.write(i, tf.maximum(std_hat, std_min))
            post_z, post_std = self.update(a_prev, z_hat, std_hat, C.read(i))
            z_prev = post_z
            std_prev = post_std
            post_z_arr = post_z_arr.write(i, post_z)
            post_std_arr = post_std_arr.write(i, post_std)
            if tf.reduce_any(tf.linalg.eigvalsh(post_std) < 0):
                print("NO SMOOTH : eigen < 0")
                #print(post_std)


        last_z_filt = post_z_arr.read(self.seq_size-1)
        last_std_filt = post_std_arr.read(self.seq_size-1)

        prev_z_smooth = post_z_arr.read(self.seq_size-1)
        prev_std_smooth = post_std_arr.read(self.seq_size-1)



        z_smooth_arr = tf.TensorArray(tf.float64, size=self.seq_size - 1, clear_after_read=False)
        std_smooth_arr = tf.TensorArray(tf.float64, size=self.seq_size - 1, clear_after_read=False)
        # tf.reverse(self.A, axis=0)
        for j in tf.range(self.seq_size-1, 0, -1):
            # print(post_std_arr.read(i).shape)
            # print(tf.transpose(A.read(i-1), perm=[0, 2, 1]).shape)
            D = tf.matmul(post_std_arr.read(j-1), tf.transpose(A.read(j), perm=[0, 2, 1]))
            # print(D.shape)
            # print(std_hat_arr.read(i-1).shape)
            #tf.print(std_hat_arr.read(j))
            D = tf.matmul(D, tf.linalg.inv(std_hat_arr.read(j), name="INVERSE_STD"))
            # print(D.shape)
            # print(prev_z_smooth.shape)
            # print(z_hat_arr.read(i-1).shape)
            temp = tf.squeeze(tf.matmul(D, tf.expand_dims(prev_z_smooth - z_hat_arr.read(j), -1)))
            z_smooth = post_z_arr.read(j-1) + temp
            temp = tf.matmul(D, (prev_std_smooth - std_hat_arr.read(j)))
            std_smooth = post_std_arr.read(j-1) + tf.matmul(temp, tf.transpose(D, perm=[0, 2, 1]))
            if tf.reduce_any(tf.linalg.eigvalsh(std_smooth) < 0):
                print("AFTER SMOOTH : eigen < 0")

            prev_z_smooth = z_smooth
            prev_std_smooth = std_smooth
            z_smooth_arr = z_smooth_arr.write(j-1, z_smooth)
            std_smooth_arr = std_smooth_arr.write(j-1, std_smooth)
        # tf.reverse(self.A, axis=0)

        return z_smooth_arr, std_smooth_arr, a_arr, A, C, last_z_filt, last_std_filt

    def get_elbo(self, images):
        z0 = tf.zeros((self.batch_size, self.dim_z), dtype=tf.float64)
        std0 = tf.eye(self.dim_z, batch_shape=[self.batch_size], dtype=tf.float64)*20.0  # z*z
        a0 = tf.zeros((self.batch_size, 1, self.dim_a), dtype=tf.float64)


        z_smooth_arr, std_smooth_arr, a_arr, A, C, last_z, last_std = self.smooth(images, z0, std0, a0)
        #_, A = tf.split(tf.transpose(A.stack(), [1, 0, 2, 3]), num_or_size_splits=[1, self.seq_size-1], axis=1)
        #C, _ = tf.split(tf.transpose(C.stack(), [1, 0, 2, 3]), num_or_size_splits=[self.seq_size-1, 1], axis=1)
        #a_arr, _ = tf.split(tf.transpose(a_arr.stack(), [1, 0, 2]), num_or_size_splits=[self.seq_size - 1, 1], axis=1)
        A = tf.transpose(A.stack(), [1, 0, 2, 3])
        C = tf.transpose(C.stack(), [1, 0, 2, 3])
        a_arr = tf.transpose(a_arr.stack(), [1, 0, 2])

        z_smooth_arr = tf.concat([tf.transpose(z_smooth_arr.stack(), [1, 0, 2]), tf.expand_dims(last_z, 1)], axis=1)
        std_smooth_arr = tf.concat([tf.transpose(std_smooth_arr.stack(), [1, 0, 2, 3]), tf.expand_dims(last_std, 1)], axis=1)
        #cov_matrix_smooth = tf.matmul(std_smooth_arr, tf.transpose(std_smooth_arr, [0, 1, 3, 2])) + (tf.eye(self.dim_z)*1e-10)
        #cov_matrix_smooth = tf.math.maximum(cov_matrix_smooth, 1e-4)
        #print(cov_matrix_smooth)
        #cov_matrix_smooth = tf.exp((std_smooth_arr + tf.transpose(std_smooth_arr, [0, 1, 3, 2]))/2)
        if tf.reduce_any(tf.linalg.eigvalsh(std_smooth_arr) < 0):
            print("elbo : eigen < 0")
            print(tf.reduce_min(tf.linalg.eigvalsh(std_smooth_arr)))
            s, u, v = tf.linalg.svd(std_smooth_arr)
            h = v @ tf.linalg.diag(s) @ tf.transpose(v, [0, 1, 3, 2])
            std_smooth_arr = (std_smooth_arr + tf.transpose(std_smooth_arr, [0, 1, 3, 2]) + h + tf.transpose(h, [0, 1, 3, 2]))/4.0 + (tf.eye(self.dim_z, dtype=tf.float64)*1e-6)
        if tf.reduce_any(tf.linalg.eigvalsh(std_smooth_arr) < 0):
            print("elbo : eigen < 0")
            print(tf.reduce_min(tf.linalg.eigvalsh(std_smooth_arr)))
        if tf.reduce_any(tf.linalg.eigvalsh(std_smooth_arr) == 0):
            print("elbo : eigen == 0")
        mvn_smooth = tfp.distributions.MultivariateNormalTriL(loc=z_smooth_arr, scale_tril=tf.linalg.cholesky(std_smooth_arr))
        #mvn_smooth = tfp.distributions.MultivariateNormalFullCovariance(z_smooth_arr, tf.exp(std_smooth_arr+tf.transpose(std_smooth_arr, [0, 1, 3, 2])/2))
        smooth_sample = mvn_smooth.sample()
        #return tf.reduce_mean(self.decode(tf.reshape(a_arr, [self.batch_size*(self.seq_size-1), self.dim_a])))+tf.reduce_mean(z_smooth_arr)+tf.reduce_mean(smooth_sample)
        #print("A shape: ", A.shape)
        #print("smooth_sample shape: ", smooth_sample.shape)
        z_transition = tf.squeeze(tf.matmul(A[:, :-1, :], tf.expand_dims(smooth_sample[:, :-1, :], -1)))

        mvn_transition = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(self.dim_z, dtype=tf.float64), scale_tril=tf.linalg.cholesky(self.Q))
        #mvn_transition = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(self.dim_z), self.Q)
        log_prob_transition = mvn_transition.log_prob(smooth_sample[:, 1:, :] - z_transition)

        mvn_emission = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(self.dim_a, dtype=tf.float64), scale_tril=tf.linalg.cholesky(self.R))
        #mvn_emission = tfp.distributions.MultivariateNormalFullCovariance(tf.zeros(self.dim_a), self.R)
        log_prob_emission = mvn_emission.log_prob(a_arr - tf.squeeze(tf.matmul(C, tf.expand_dims(smooth_sample, -1))))

        mvn_0 = tfp.distributions.MultivariateNormalTriL(loc=z0, scale_tril=tf.linalg.cholesky(std0))
        #mvn_0 = tfp.distributions.MultivariateNormalFullCovariance(z0, std0)
        log_prob_0 = mvn_0.log_prob(smooth_sample[:, 0])

        entropy = - mvn_smooth.log_prob(smooth_sample)

        log_probs = [tf.reduce_sum(log_prob_transition, axis=[1]),
                     tf.reduce_sum(log_prob_emission, axis=[1]),
                     log_prob_0,
                     tf.reduce_sum(entropy, axis=[1])]

        #kf_elbo = tf.reduce_sum(log_probs, axis=[2])
        #print("\nlog_prob_transition: ", tf.reduce_sum(log_prob_transition))
        #print("log_prob_emission: ", tf.reduce_sum(log_prob_emission))
        #print("log_prob_0: ", tf.reduce_sum(log_prob_0))
        #print("entropy: ", tf.reduce_sum(entropy))
        kf_elbo = tf.reduce_sum(log_prob_transition, axis=[1]) + tf.reduce_sum(log_prob_emission, axis=[1]) + log_prob_0 + tf.reduce_sum(entropy, axis=[1])

        return kf_elbo


    def get_loss(self, im):
        elbo_kf = tf.reduce_sum(self.get_elbo(im))
        mu_a, std_a = self.encode(tf.reshape(im, [self.batch_size*self.seq_size, self.im_shape, self.im_shape]))
        #tf.print(std_a.shape)
        #mvn_a = tfp.distributions.MultivariateNormalDiag(mu_a, std_a)
        #a_seq = mvn_a.sample()
        a_seq = self.reparameterize(mu_a, tf.math.log(std_a))
        #print("\na_seq shape: ", a_seq.shape)
        #print("mu_a seq shape: ", mu_a.shape)
        #print("std_a seq shape: ", std_a.shape)
        #print("a seq shape: ", a_seq.shape)
        #log_qa_x = mvn_a.log_prob(a_seq)
        log_qa_x = self.log_gaussian(a_seq, mu_a, std_a)
        log_qa_x = tf.reduce_sum(tf.reshape(log_qa_x, [self.batch_size, self.seq_size]), [1])


        #return tf.reduce_mean(elbo_kf)+tf.reduce_mean(log_qa_x)

        #a_seq = self.reparameterize(mu_a, logvar_a)
        #lnpdf = self.log_normal_pdf(a_seq, mu_a, logvar_a)
        #log_qa_x = tf.reduce_sum(tf.reshape(self.log_normal_pdf(a_seq, mu_a, logvar_a), [self.batch_size, self.seq_size]), [1])
        #return elbo_kf + tf.reduce_mean(lnpdf)
        im_logit = tf.reshape(self.decode(a_seq, True), [self.batch_size, self.seq_size, self.im_shape, self.im_shape, 1])

        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=im_logit, labels=im)
        #log_px_a = -tf.reduce_sum(cross_ent, axis=[1, 2, 3, ])
        log_px_a = self.log_bernoulli(im, im_logit, eps=1e-6)

        #print("elbo : ", tf.reduce_sum(elbo_kf))
        #print("log_px|a : ", tf.reduce_sum(log_px_a))
        #print("log qa|x : ", tf.reduce_sum(log_qa_x))

        #tf.print((elbo_kf + log_px_z - log_qa_x).shape)
        #print("\nelbo_kf :", -tf.reduce_sum(elbo_kf))
        #print("log_px_a :", -tf.reduce_sum(log_px_a))
        #print("log_qa_x :", tf.reduce_sum(log_qa_x))

        #KL = tfp.distributions.kl_divergence(mvn_a, tfp.distributions.MultivariateNormalTriL(0, tf.linalg.cholesky(tf.eye(self.dim_a, batch_shape=[self.batch_size*self.seq_size], dtype=tf.float64))))

        loss = -tf.reduce_sum(elbo_kf + log_px_a - log_qa_x)
        #loss = tf.reduce_sum(-elbo_kf + log_px_a - KL)

        return loss

    def get_accuracy(self, im):

        z0 = tf.zeros((self.batch_size, self.dim_z), dtype=tf.float64)
        std0 = tf.eye(self.dim_z, batch_shape=[self.batch_size], dtype=tf.float64)*20.0  # z*z
        a0 = tf.zeros((self.batch_size, 1, self.dim_a), dtype=tf.float64)

        z_smooth, std_smooth, a_arr, A, C, last_z, last_std = self.smooth(im, z0, std0, a0)
        a_arr.mark_used()
        A.mark_used()

        #z_smooth_arr = tf.transpose(z_smooth.stack(), [1, 0, 2])
        #std_smooth_arr = tf.transpose(std_smooth.stack(), [1, 0, 2, 3])
        #cov_matrix_smooth = tf.sqrt(tf.matmul(std_smooth_arr, tf.transpose(std_smooth_arr, [0, 1, 3, 2])))
        #mvn_smooth = tfp.distributions.MultivariateNormalTriL(loc=z_smooth_arr, scale_tril=tf.linalg.cholesky(cov_matrix_smooth))

        z = tf.concat([tf.transpose(z_smooth.stack(), [1, 0, 2]), tf.expand_dims(last_z, 1)], 1)
        std = tf.concat([tf.transpose(std_smooth.stack(), [1, 0, 2, 3]), tf.expand_dims(last_std, 1)], 1)
        #std = tf.matmul(std, tf.transpose(std, [0, 1, 3, 2])) + (tf.eye(self.dim_z)*1e-10)
        #std = tf.exp((std + tf.transpose(std, perm=[0, 1, 3, 2]))/2)
        #print("z shape: ", z.shape)
        #print("std shape: ", std.shape)
        #print("std :", std[0, 19, :, :])
        #print("std min : ", tf.reduce_min(std))
        #std = tf.math.maximum(std, 1e-4)
        if tf.reduce_any(tf.linalg.eigvalsh(std) < 0):
            print("acc : eigen < 0")
            print(tf.reduce_min(tf.linalg.eigvalsh(std)))
            s, u, v = tf.linalg.svd(std)
            print(v.shape)
            print(s.shape)
            print(tf.linalg.diag(s).shape)
            h = v @ tf.linalg.diag(s) @ tf.transpose(v, [0, 1, 3, 2])
            std = (std + tf.transpose(std, [0, 1, 3, 2]) + h + tf.transpose(h, [0, 1, 3, 2]))/4.0 + (tf.eye(self.dim_z, dtype=tf.float64)*1e-6)
        if tf.reduce_any(tf.linalg.eigvalsh(std) < 0):
            print("acc : eigen < 0")
            print(tf.reduce_min(tf.linalg.eigvalsh(std)))
        if tf.reduce_any(tf.linalg.eigvalsh(std) == 0):
            print("acc : eigen == 0")
        cholesky = tf.linalg.cholesky(std)
        mvn = tfp.distributions.MultivariateNormalTriL(loc=z, scale_tril=cholesky)
        #mvn = tfp.distributions.MultivariateNormalFullCovariance(z, std)
        samples = mvn.sample()


        a = tf.squeeze(tf.matmul(tf.transpose(C.stack(), [1, 0, 2, 3]), tf.expand_dims(samples, -1)))
        a = tf.reshape(a, [self.batch_size*self.seq_size, self.dim_a])



        #mu_a, logvar_a = self.encode(tf.reshape(im, [self.batch_size*self.seq_size, img_size, img_size, 1]))
        #a = model.reparameterize(mu_a, logvar_a)
        im_logit = tf.reshape(self.decode(a, True), [self.batch_size, self.seq_size, self.im_shape, self.im_shape, 1])
        accuracy = tf.reduce_mean(tf.square(im_logit - im))

        return accuracy

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, a):
        a_inf = self.inference_net(a)
        # tf.print("x_inf : ", x_inf[0])
        mean, std = tf.split(a_inf, num_or_size_splits=[self.dim_a, self.dim_a], axis=1)
        std = tf.nn.sigmoid(std)
        return mean, std

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape, dtype=tf.float64)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def log_bernoulli(self, x, p, eps=0.0):
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return x * tf.math.log(p) + (1 - x) * tf.math.log(1 - p)

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
      log2pi = tf.math.log(2. * np.pi)
      return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def log_gaussian(self, x, mean, var):
        const_log_pdf = (- 0.5 * np.log(2 * np.pi)).astype('float64')
        # return - 0.5 * tf.log(2*np.pi) - var / 2 - tf.square((x - mean)) / (2 * tf.exp(var) + 1e-8)
        return const_log_pdf - tf.math.log(var) / 2 - tf.square(x - mean) / (2 * var)

    def compute_loss(self, x):
        loss = self.get_loss(x)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=False)

        x_obs = self.reparameterize(x_logit, logvar=tf.math.log(self.std))


        #reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
        #  tf.keras.layers.Flatten()(x_logit), scale_identity_multiplier=0.05).log_prob(tf.keras.layers.Flatten()(x)))

        #kl_divergence = tf.reduce_sum(tf.keras.metrics.kullback_leibler_divergence(x, x_logit), axis=[1, 2])

        #cross_ent = self._gaussian_log_likelihood(x_logit, mean, logvar)
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

        #reconstr = tf.reduce_mean(tf.square(x - x_logit))
        reconstr = self._gaussian_log_likelihood(x, x_obs, std=self.std)
        kl = self._kl_diagnormal_stdnormal(mean, logvar)
        return tf.reduce_mean(reconstr + kl)

        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
        #logpx_z = tf.reduce_mean(tf.square(x - x_logit))
        #logpz = self.log_normal_pdf(z, 0., 0.)
        #logqz_x = self.log_normal_pdf(z, mean, logvar)
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x)
        #return tf.reduce_mean(reconstruction_term + kl_divergence)
        #return tf.reduce_sum(tf.square(x - x_logit))
        """
        return loss

    def reconstruct(self, imgs):
        z0 = tf.zeros((self.batch_size, self.dim_z), dtype=tf.float64)
        std0 = tf.eye(self.dim_z, batch_shape=[self.batch_size], dtype=tf.float64)*20.0  # z*z
        a0 = tf.zeros((self.batch_size, 1, self.dim_a), dtype=tf.float64)

        z_smooth, std_smooth, a_arr, A, C, last_z, last_std = self.smooth(imgs, z0, std0, a0)
        #a_arr.mark_used()

        A.mark_used()
        C.mark_used()
        z_smooth.mark_used()
        std_smooth.mark_used()


        #a_arr = tf.transpose(a_arr.stack(), [1, 0, 2])

        #z = tf.concat([tf.transpose(z_smooth.stack(), [1, 0, 2]), tf.expand_dims(last_z, 1)], 1)
        #std = tf.concat([tf.transpose(std_smooth.stack(), [1, 0, 2, 3]), tf.expand_dims(last_std, 1)], 1)
        #if tf.reduce_any(tf.linalg.eigvalsh(std) < 0):
        #    s, u, v = tf.linalg.svd(std)
        #    h = v @ tf.linalg.diag(s) @ tf.transpose(v, [0, 1, 3, 2])
        #    std = (std + tf.transpose(std, [0, 1, 3, 2]) + h + tf.transpose(h, [0, 1, 3, 2])) / 4.0 + (tf.eye(self.dim_z, dtype=tf.float64) * 1e-6)

        #mvn = tfp.distributions.MultivariateNormalTriL(z, tf.linalg.cholesky(std))
        #samples = mvn.sample()

        #a = tf.squeeze(tf.matmul(tf.transpose(C.stack(), [1, 0, 2, 3]), tf.expand_dims(samples, -1)))
        #a = tf.reshape(a, [self.batch_size * self.seq_size, self.dim_a])

        #a_arr = tf.reshape(a_arr, [self.batch_size*self.seq_size, self.dim_a])

        mu_a, logvar_a = self.encode(tf.reshape(imgs, [self.batch_size*self.seq_size, self.im_shape, self.im_shape, 1]))
        a = self.reparameterize(mu_a, logvar_a)
        #print("decode a_arr :", self.decode(a_arr, True).shape)
        im_logit = tf.reshape(self.decode(a, True), [self.batch_size, self.seq_size, self.im_shape, self.im_shape, 1])
        return im_logit

    def predict_seq(self, imgs, mask):
        z0 = tf.zeros((self.batch_size, self.dim_z), dtype=tf.float64)
        std0 = tf.eye(self.dim_z, batch_shape=[self.batch_size], dtype=tf.float64)  # z*z
        a0 = tf.zeros((self.batch_size, 1, self.dim_a), dtype=tf.float64)


    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compute_accuracy(self, x):
        """mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z, apply_sigmoid=True)
        accuracy = tf.reduce_mean(tf.square(x_logit - x))"""
        return self.get_accuracy(x)