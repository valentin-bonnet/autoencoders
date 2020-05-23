import tensorflow as tf

class Memory(tf.keras.layers.Layer):
    def __init__(self, unit=100, decay=0.8, threshold=0.1*1000,  k=256, c=3, kernel=15, **kwargs):
        self.m = unit
        self.decay = decay
        self.k_shape = k
        self.v_shape = c
        self.kernel = kernel
        self.threshold = threshold*((kernel*kernel)/100.0)
        self.hw_shape = 64*64
        self.batch_shape = 2
        #self.lstm = tf.keras.Sequential()
        #self.lstm.add(tf.keras.layers.Input(shape=(top_a+unit, k),batch_size=4))
        #self.lstm.add(tf.keras.layers.LSTM(self.m+self.top_a, stateful=True))
        self.state_size = [tf.TensorShape([self.m, self.k_shape]), tf.TensorShape([self.m, self.v_shape]), tf.TensorShape([self.m])]
        self.output_size = [tf.TensorShape([self.m, self.k_shape]), tf.TensorShape([self.m, self.v_shape])]

        super(Memory, self).__init__(**kwargs)

    def build(self, input_shape):
        # input : [(batch size, HW, K), (batch size, HW, V), (batch size, HW, 1)]
        self.batch_shape = input_shape[0][0]
        self.hw_shape = input_shape[0][1]
        #self.a_shape = input_shape[0][3]
        self.m_k = self.add_weight(shape=(self.batch_shape, self.m, self.kernel**2, self.k_shape), initializer='zeros', trainable=False, name='mk')
        self.m_v = self.add_weight(shape=(self.batch_shape, self.m, self.kernel**2, self.v_shape), initializer='zeros', trainable=False, name='mv')
        self.m_u = self.add_weight(shape=(self.batch_shape, self.m), initializer='ones', trainable=False, name='mu')


        #self.wf = self.add_weight(shape=(self.m, self.m+self.a_shape), initializer='random_normal', trainable=True, name='wf')
        #self.bf = self.add_weight(shape=(self.m, ), initializer='zeros', trainable=True, name='bf')

        #self.wi = self.add_weight(shape=(self.m, self.hw_shape), initializer='random_normal', trainable=True, name='wi')

    def call(self, inputs, **kwargs):

        k, v, rkn_score = tf.nest.flatten(inputs)  # [(bs, HW, K), (bs, HW, V), (bs, HW, 1)]
        #self.m_k, self.m_v, self.m_u= tf.nest.flatten(states)
        # m_k = states[0] # [(bs, m, K), (bs, m, V)]
        # m_v = states[1] # [(bs, m, K), (bs, m, V)]
        # m_u = states[2] # [(bs, m, K), (bs, m, V)]
        idx = tf.argsort(self.m_u, axis=-1, direction='ASCENDING', name=None)
        m_u_sorted = tf.gather(self.m_u, idx, batch_dims=1, axis=1)
        m_k_sorted = tf.gather(self.m_k, idx, batch_dims=1, axis=1)
        m_v_sorted = tf.gather(self.m_v, idx, batch_dims=1, axis=1)

        #k_patch = tf.image.extract_patches(images=tf.reshape(k, [-1, 64, 64, 256]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.kernel, self.kernel, 1], rates=[1, 1, 1, 1], padding="VALID")
        #v_patch = tf.image.extract_patches(images=tf.reshape(v, [-1, 64, 64, 3]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.kernel, self.kernel, 1], rates=[1, 1, 1, 1], padding="VALID")
        #k_patch = tf.reshape(k_patch, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel*self.kernel*256])
        #v_patch = tf.reshape(v_patch, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel*self.kernel*3])
        k_patch = tf.image.extract_patches(images=tf.reshape(k, [-1, 64, 64, 256]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
        v_patch = tf.image.extract_patches(images=tf.reshape(v, [-1, 64, 64, 3]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
        k_patch = tf.reshape(k_patch, [self.batch_shape, self.hw_shape, self.kernel*self.kernel*256])
        v_patch = tf.reshape(v_patch, [self.batch_shape, self.hw_shape, self.kernel*self.kernel*3])
        s = tf.nn.softmax(k_patch@ tf.transpose(tf.reshape(self.m_k, [self.batch_shape, self.m, self.kernel*self.kernel*256]), [0, 2, 1])) # (bs, nb_patch, size_patch*256) @ (bs, m, size_patch*256) = (bs, nb_patch, m)
        max_s_hw = tf.reduce_max(s, axis=-1)  # (bs, nb_patch)
        max_s_m = tf.reduce_max(s, axis=-2)  # (bs, M)
        wv_bool = tf.where(max_s_hw < self.threshold, True, False)  # (bs, top)
        all_ones = tf.ones_like(max_s_hw)

        idx = tf.argsort(max_s_hw, axis=-1, direction='ASCENDING', name=None)
        k_sorted = tf.gather(k_patch, idx, batch_dims=1, axis=1)
        v_sorted = tf.gather(v_patch, idx, batch_dims=1, axis=1)
        #k_sorted = tf.reshape(k_sorted, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel**2, self.k_shape])
        #v_sorted = tf.reshape(v_sorted, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel**2, self.v_shape])
        k_sorted = tf.reshape(k_sorted, [self.batch_shape, self.hw_shape, self.kernel**2, self.k_shape])
        v_sorted = tf.reshape(v_sorted, [self.batch_shape, self.hw_shape, self.kernel**2, self.v_shape])
        rkn_score_sorted = tf.gather(rkn_score, idx, batch_dims=1, axis=1)


        write_ones = tf.ragged.boolean_mask(all_ones, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m])
        write_k = tf.ragged.boolean_mask(k_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m, self.kernel*self.kernel, self.k_shape])
        write_v = tf.ragged.boolean_mask(v_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m, self.kernel*self.kernel, self.v_shape])
        write_rkn_score = tf.ragged.boolean_mask(rkn_score_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m])

        self.m_u = (self.decay * m_u_sorted + max_s_m) * (1 - write_ones) + write_ones + write_rkn_score # (bs, m)
        write_ones = tf.expand_dims(tf.expand_dims(write_ones, -1), -1)
        self.m_k = m_k_sorted*(1. - write_ones) + write_k # (bs, m, kernel**2, k)
        self.m_v = m_v_sorted*(1. - write_ones) + write_v # (bs, m, kernel**2, v)



        return [self.m_k, self.m_v]

    def call_init(self, inputs, bs):

        k, v, rkn_score = tf.nest.flatten(inputs)  # [(bs, HW, K), (bs, HW, V), (bs, HW, 1)]
        #self.m_k, self.m_v, self.m_u= tf.nest.flatten(states)
        # m_k = states[0] # [(bs, m, K), (bs, m, V)]
        # m_v = states[1] # [(bs, m, K), (bs, m, V)]
        # m_u = states[2] # [(bs, m, K), (bs, m, V)]
        idx = tf.argsort(self.m_u, axis=-1, direction='ASCENDING', name=None)
        m_u_sorted = tf.gather(self.m_u, idx, batch_dims=1, axis=1)
        m_k_sorted = tf.gather(self.m_k, idx, batch_dims=1, axis=1)
        m_v_sorted = tf.gather(self.m_v, idx, batch_dims=1, axis=1)

        # k_patch = tf.image.extract_patches(images=tf.reshape(k, [-1, 64, 64, 256]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.kernel, self.kernel, 1], rates=[1, 1, 1, 1], padding="VALID")
        # v_patch = tf.image.extract_patches(images=tf.reshape(v, [-1, 64, 64, 3]), sizes=[1, self.kernel, self.kernel, 1], strides=[1, self.kernel, self.kernel, 1], rates=[1, 1, 1, 1], padding="VALID")
        # k_patch = tf.reshape(k_patch, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel*self.kernel*256])
        # v_patch = tf.reshape(v_patch, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel*self.kernel*3])
        k_patch = tf.image.extract_patches(images=tf.reshape(k, [-1, 64, 64, 256]),
                                           sizes=[1, self.kernel, self.kernel, 1], strides=[1, 1, 1, 1],
                                           rates=[1, 1, 1, 1], padding="SAME")
        v_patch = tf.image.extract_patches(images=tf.reshape(v, [-1, 64, 64, 3]),
                                           sizes=[1, self.kernel, self.kernel, 1], strides=[1, 1, 1, 1],
                                           rates=[1, 1, 1, 1], padding="SAME")
        k_patch = tf.reshape(k_patch, [self.batch_shape, self.hw_shape, self.kernel * self.kernel * 256])
        v_patch = tf.reshape(v_patch, [self.batch_shape, self.hw_shape, self.kernel * self.kernel * 3])
        s = tf.nn.softmax(k_patch@ tf.transpose(tf.reshape(self.m_k, [self.batch_shape, self.m, self.kernel*self.kernel*256]), [0, 2, 1])) # (bs, nb_patch, size_patch*256) @ (bs, m, size_patch*256) = (bs, nb_patch, m)
        max_s_hw = tf.reduce_max(s, axis=-1)  # (bs, nb_patch)
        max_s_m = tf.reduce_max(s, axis=-2)  # (bs, M)
        wv_bool = tf.where(max_s_hw < self.threshold, True, False)  # (bs, top)
        all_ones = tf.ones_like(max_s_hw)

        idx = tf.argsort(max_s_hw, axis=-1, direction='ASCENDING', name=None)
        k_sorted = tf.gather(k_patch, idx, batch_dims=1, axis=1)
        v_sorted = tf.gather(v_patch, idx, batch_dims=1, axis=1)
        # k_sorted = tf.reshape(k_sorted, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel**2, self.k_shape])
        # v_sorted = tf.reshape(v_sorted, [self.batch_shape, (64//self.kernel)*(64//self.kernel), self.kernel**2, self.v_shape])
        k_sorted = tf.reshape(k_sorted, [self.batch_shape, self.hw_shape, self.kernel ** 2, self.k_shape])
        v_sorted = tf.reshape(v_sorted, [self.batch_shape, self.hw_shape, self.kernel ** 2, self.v_shape])
        rkn_score_sorted = tf.gather(rkn_score, idx, batch_dims=1, axis=1)


        write_ones = tf.ragged.boolean_mask(all_ones, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m])
        write_k = tf.ragged.boolean_mask(k_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m, self.kernel*self.kernel, self.k_shape])
        write_v = tf.ragged.boolean_mask(v_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m, self.kernel*self.kernel, self.v_shape])
        write_rkn_score = tf.ragged.boolean_mask(rkn_score_sorted, wv_bool).to_tensor(default_value=0., shape=[self.batch_shape, self.m])

        self.m_u = (self.decay * m_u_sorted + max_s_m) * (1 - write_ones) + write_ones + write_rkn_score # (bs, m)
        write_ones = tf.expand_dims(tf.expand_dims(write_ones, -1), -1)
        self.m_k = m_k_sorted*(1. - write_ones) + write_k # (bs, m, kernel**2, k)
        self.m_v = m_v_sorted*(1. - write_ones) + write_v # (bs, m, kernel**2, v)


    """
    def call(self, inputs, states):        
        m_k, m_v = tf.nest.flatten(states)
        attention, k, v = tf.nest.flatten(inputs)  # [(bs, H, W, A), (bs, H, W, K), (bs, H, W, V)]
        attention = tf.reshape(attention, [self.batch_shape, self.a_shape, self.hw_shape])  # (bs, A, HW)
        attention = tf.reduce_sum(tf.nn.softmax(attention, -1), -2) # (bs, HW)
        _, top_indx = tf.math.top_k(attention, k=self.top_a, sorted=False)

        attention_k = tf.gather(tf.reshape(k, [self.batch_shape, self.hw_shape, self.k_shape]), top_indx, axis=1, batch_dims=1)  # (bs, top_a, k)

        attention_k = tf.reshape(attention_k, [self.batch_shape, self.top_a, self.k_shape])

        attention_v = tf.gather(tf.reshape(v, [self.batch_shape, self.hw_shape, self.v_shape]), top_indx, axis=1, batch_dims=1)  # (bs, top_a, v)
        attention_v = tf.reshape(attention_v, [self.batch_shape, self.top_a, self.v_shape])

        k_mk = tf.concat([attention_k, m_k], 1)
        v_mv = tf.concat([attention_v, m_v], 1)
        score = self.lstm(k_mk)
        _, top_idx_m = tf.math.top_k(score, k=self.m, sorted=False)
        m_k = tf.gather(k_mk, top_idx_m, axis=1, batch_dims=1)
        m_v = tf.gather(v_mv, top_idx_m, axis=1, batch_dims=1)

        return [m_k, m_v], [m_k, m_v]  # inputs, states"""

    """def call(self, inputs):
        attention, k, v = tf.nest.flatten(inputs) # [(bs, H, W, A), (bs, H, W, K), (bs, H, W, V)]
        attention = tf.reshape(attention, [self.batch_shape, self.a_shape, self.hw_shape]) # (bs, A, HW)
        k = tf.reshape(k, [self.batch_shape, self.hw_shape, self.k_shape]) # (bs, HW, K)
        v = tf.reshape(v, [self.batch_shape, self.hw_shape, self.v_shape]) # (bs, HW, V)
        attention_k = attention @ k # (bs, A, K)
        forget_input = tf.transpose(tf.concat([self.m_k, attention_k], -2), [0, 2, 1]) # (bs, K, M+A)
        forget_gate = self.wf @ tf.expand_dims(forget_input, -1) # (bs, K, M, 1)
        forget_gate = tf.reshape(forget_gate, [-1, self.k_shape, self.m]) + self.bf # (bs, K, M)
        forget_gate = tf.transpose(forget_gate, [0, 2, 1]) # (bs, M, K)
        forget_gate = tf.reduce_sum(forget_gate, -1)  # (bs, M)
        forget_gate = tf.sigmoid(forget_gate)
        forget_gate = tf.expand_dims(forget_gate, -1)
        self.m_k = forget_gate * self.m_k + (1 - forget_gate) * (self.wi @ k) # (bs, M, K)
        self.m_v = forget_gate * self.m_v + (1 - forget_gate) * (self.wi @ v) # (bs, M, V)

        self.m_k = tf.reshape(self.m_k, [-1, self.m, self.k_shape])
        self.m_v = tf.reshape(self.m_v, [-1, self.m, self.v_shape])

        return [self.m_k, self.m_v], #inputs, states"""

    """
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        self.m_k = tf.zeros_like(self.m_k)
        self.m_v = tf.zeros_like(self.m_v)
        self.m_u = tf.ones_like(self.m_u)
        return [self.m_k, self.m_v, self.m_u]
    """

    def get_init_state(self, bs):
        self.m_k = tf.zeros([bs, self.m, self.kernel**2, self.k_shape])
        self.m_v = tf.zeros([bs, self.m, self.kernel**2, self.v_shape])
        self.m_u = tf.ones([bs, self.m])
        self.threshold = 100.

    def get_config(self):
        return {'units': self.m}