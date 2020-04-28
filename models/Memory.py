import tensorflow as tf

class Memory(tf.keras.layers.Layer):
    def __init__(self, unit=100, **kwargs):
        self.m = unit
        super(Memory, self).__init__(**kwargs)

    def build(self, input_shape):
        # input : [(batch size, H, W, A), (batch size, H, W, K), (batch size, H, W, V)]
        self.batch_shape = input_shape[0][0]
        self.hw_shape = input_shape[0][1] * input_shape[0][2]
        self.a_shape = input_shape[0][3]
        self.k_shape = input_shape[2][3]
        self.v_shape = input_shape[3][3]

        self.wf = self.add_weight(shape=(self.m, self.m+self.a_shape), initializer='random_normal', trainable=True)
        self.bf = self.add_weight(shape=(self.m, ), initializer='zeros', trainable=True)

        self.wi = self.add_weight(shape=(self.m, self.hw_shape), initializer='random_normal', trainable=True)

    def call(self, inputs, states):
        m_k, m_v = tf.nest.flatten(states)
        attention, k, v = tf.nest.flatten(inputs) # [(bs, H, W, A), (bs, H, W, K), (bs, H, W, V)]
        attention = tf.reshape(attention, [self.batch_shape, self.a_shape, self.hw_shape]) # (bs, A, HW)
        k = tf.reshape(k, [self.batch_shape, self.hw_shape, self.k_shape]) # (bs, HW, K)
        v = tf.reshape(v, [self.batch_shape, self.hw_shape, self.v_shape]) # (bs, HW, V)
        attention_k = attention @ k # (bs, A, K)
        forget_input = tf.transpose(tf.concat([self.m_k, attention_k], -2), [0, 2, 1]) # (bs, K, M+A)
        forget_gate = tf.sigmoid(self.wf @ forget_input + self.bf) # (bs, M)

        m_k = forget_gate * m_k + (1 - forget_gate) * (self.wi @ k) # (bs, M, K)
        m_v = forget_gate * m_v + (1 - forget_gate) * (self.wi @ v) # (bs, M, V)

        return [m_k, m_v], [m_k, m_v] #inputs, states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        m_k = self.add_weight(shape=(batch_size, self.m, self.k_shape), initializer='zeros', trainable=False)
        m_v = self.add_weight(shape=(batch_size, self.m, self.v_shape), initializer='zeros', trainable=False)
        return [m_k, m_v]

    def get_config(self):
        return {'units': self.m}