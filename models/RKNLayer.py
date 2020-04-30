import tensorflow as tf

class RKNLayer(tf.keras.layers.Layer):
    def __init__(self, m, k, b, alpha_unit, **kwargs):
        self.K = k
        self.B = b
        self.alpha_unit = alpha_unit
        self.M = m
        self.N = m*2
        self.state_size = [m, m, m, m]
        self.output_size = m
        super(RKNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # [(bs, m)]
        self.batch_size = input_shape[0]
        init_std_trans_u = tf.ones((self.batch_size, self.M)) * 1.1
        init_std_trans_l = tf.ones((self.batch_size, self.M)) * 1.1

        B11_init = tf.eye(self.M, batch_shape=[self.K])
        B12_init = tf.eye(self.M, batch_shape=[self.K]) * 0.2
        B21_init = tf.eye(self.M, batch_shape=[self.K]) * -0.2
        B22_init = tf.eye(self.M, batch_shape=[self.K])

        self.std_trans_u = tf.exp(tf.Variable(initial_value=init_std_trans_u, trainable=True, name='std_trans_u'))
        self.std_trans_l = tf.exp(tf.Variable(initial_value=init_std_trans_l, trainable=True, name='std_trans_l'))
        self.B11 = tf.linalg.band_part(tf.Variable(initial_value=B11_init, trainable=True, name='B11'), self.B, self.B)
        self.B12 = tf.linalg.band_part(tf.Variable(initial_value=B12_init, trainable=True, name='B12'), self.B, self.B)
        self.B21 = tf.linalg.band_part(tf.Variable(initial_value=B21_init, trainable=True, name='B21'), self.B, self.B)
        self.B22 = tf.linalg.band_part(tf.Variable(initial_value=B22_init, trainable=True, name='B22'), self.B, self.B)

        self._lgssm_parameters_inference = tf.keras.Sequential()
        self._lgssm_parameters_inference.add(tf.keras.layers.Input(shape=(None, self.N), batch_size=self.batch_size))
        self._lgssm_parameters_inference.add(tf.keras.layers.LSTM(self.alpha_unit))
        self._lgssm_parameters_inference.add(tf.keras.layers.Flatten())
        self._lgssm_parameters_inference.add(tf.keras.layers.Dense(self.K, activation=tf.nn.softmax))


    def _pred(self, z_post, std_u, std_l, std_s):
        alpha = self._lgssm_parameters_inference(tf.expand_dims(z_post, 1)) # (bs, K)
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

    def _update(self, z_prior, std_u_prior, std_l_prior, std_s_prior, a_mean, a_std):
        q_u = std_u_prior / (std_u_prior + a_std)
        q_l = std_s_prior / (std_u_prior + a_std)
        residual = a_mean - z_prior[:, :self.M]

        z_post = z_prior + tf.concat([q_u*residual, q_l*residual], -1)

        std_u_post = (1 - q_u) * std_u_prior
        std_l_post = std_l_prior - (q_u * std_s_prior)
        std_s_post = (1 - q_u) * std_s_prior

        return z_post, std_u_post, std_l_post, std_s_post

    def call(self, inputs, states):
        # (bs, M)
        #print("RKNLayer inputs shape: ", inputs.shape)
        a_mean, a_std = tf.split(inputs, num_or_size_splits=[self.M, self.M], axis=-1)
        z, std_u, std_l, std_s = tf.nest.flatten(states)
        z_prior, std_u_prior, std_l_prior, std_s_prior = self._pred(z, std_u, std_l, std_s)
        z_post, std_u_post, std_l_post, std_s_post = self._update(z_prior, std_u_prior, std_l_prior, std_s_prior, a_mean, a_std)
        return z_post, [z_post, std_u_post, std_l_post, std_s_post]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        init_z0 = tf.zeros((batch_size, self.N))
        z0 = tf.Variable(initial_value=init_z0, trainable=False)
        init_std_u = tf.ones((batch_size, self.M)) * 10.0  # z*z
        init_std_l = tf.ones((batch_size, self.M)) * 10.0  # z*z
        init_std_s = tf.zeros((batch_size, self.M))
        std_u = tf.Variable(initial_value=init_std_u, trainable=False)
        std_l = tf.Variable(initial_value=init_std_l, trainable=False)
        std_s = tf.Variable(initial_value=init_std_s, trainable=False)
        return [z0, std_u, std_l, std_s]

    def get_config(self):
        return {'k': self.K, 'b': self.B, 'alpha_unit': self.alpha_unit}