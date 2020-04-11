import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class RKN(tf.keras.Model):
    def __init__(self, layers=[32, 32, 32], latent_dim=256, input_shape=64, sequence_length=20, M=100, use_bn=False):
        super(RKN, self).__init__()
        self.model_type = 'RKN'
        self.batch_size = 8
        self.architecture = layers.copy()
        self.latent_dim = latent_dim
        self.use_bn = use_bn
        self.seq_size = sequence_length
        self.im_shape = input_shape
        self.B = 1
        self.K = 15
        self.M = M # Observation space (Latent space)
        self.N = 2*M # Z space

        str_arch = '_'.join(str(x) for x in self.architecture)
        str_bn = 'BN' if use_bn else ''
        self.description = '_'.join(filter(None, ['KVAE', str_arch, 'lat' + str(self.M), 'seq' + str(self.seq_size), str_bn]))

        H0 = tf.eye(self.M, batch_shape=[self.batch_size])
        H1 = tf.zeros((self.batch_size, self.M))
        self.H = tf.concat([H0, H1], axis=1)

        init_z0 = tf.zeros((self.batch_size, self.N))
        init_std_u = tf.ones((self.batch_size, self.N/2)) * 10.0  # z*z
        init_std_l = tf.ones((self.batch_size, self.N/2))* 10.0  # z*z
        init_std_s = tf.zeros((self.batch_size, self.N/2))
        B11_init = tf.eye(self.M, batch_shape=[self.batch_size, self.K])
        B12_init = tf.eye(self.M, batch_shape=[self.batch_size, self.K]) * 0.2
        B21_init = tf.eye(self.M, batch_shape=[self.batch_size, self.K]) * -0.2
        B22_init = tf.eye(self.M, batch_shape=[self.batch_size, self.K])

        self.z0 = tf.Variable(initial_value=init_z0, trainable=False)
        self.std_u = tf.Variable(initial_value=init_std_u, trainable=False)
        self.std_l = tf.Variable(initial_value=init_std_l, trainable=False)
        self.std_s = tf.Variable(initial_value=init_std_s, trainable=False)
        self.B11 = tf.linalg.band_part(tf.Variable(initial_value=B11_init), self.B, self.B)
        self.B12 = tf.linalg.band_part(tf.Variable(initial_value=B12_init), self.B, self.B)
        self.B21 = tf.linalg.band_part(tf.Variable(initial_value=B21_init), self.B, self.B)
        self.B22 = tf.linalg.band_part(tf.Variable(initial_value=B22_init), self.B, self.B)


        self.lgssm_parameters_inference = tf.keras.Sequential()
        self.lgssm_parameters_inference.add(tf.keras.layers.Input(shape=(None, self.N), batch_size=self.batch_size))
        self.lgssm_parameters_inference.add(tf.keras.layers.LSTM(100, stateful=True))
        self.lgssm_parameters_inference.add(tf.keras.layers.Flatten())
        self.lgssm_parameters_inference.add(tf.keras.layers.Dense(self.K, activation=tf.nn.softmax))

        ## ENCODER
        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 1)))
        for l in layers:
            self.inference_net.add(tf.keras.layers.Conv2D(filters=l, kernel_size=5, strides=1, padding='same'))
            if use_bn:
                self.inference_net.add(tf.keras.layers.BatchNormalization())
            self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(self.M))

        ## DECODER

        layers.reverse()
        # size_decoded_frame = int(input_shape//(2**len(layers)))
        size_decoded_frame = self.im_shape
        size_decoded_layers = int(layers[0] // 2)

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(self.M,)))
        self.generative_net.add(tf.keras.layers.Dense(size_decoded_frame * size_decoded_frame * size_decoded_layers))
        self.generative_net.add(
            tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))

        for l in layers:
            self.generative_net.add(
                tf.keras.layers.Conv2DTranspose(filters=l, kernel_size=5, strides=1, padding='same'))
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
        alpha = self.lgssm_parameters_inference(z_post) # (bs, K)
        B11 = alpha @ self.B11 # (bs, N, N)
        B12 = alpha @ self.B12 # (bs, N, N)
        B21 = alpha @ self.B21 # (bs, N, N)
        B22 = alpha @ self.B22 # (bs, N, N)
        A_pred = tf.concat([tf.concat([B11, B12], -1),
                            tf.concat([B21, B22], -1)], -2)
        z_prior = A_pred @ z_post
        std_u =





