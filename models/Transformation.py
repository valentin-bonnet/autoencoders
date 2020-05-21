import tensorflow as tf
import numpy as np

class Transformation(tf.keras.layers.Layer):
    def __init__(self, p=0.5, p_seq=0.2, **kwargs):
        self.p = p
        self.p_seq = p_seq
        super(Transformation, self).__init__(**kwargs)

    def build(self, input_shape):
        # input : (batch size, t, h, w, ck)
        self.batch_shape = input_shape[0]
        self.seq_size = input_shape[1]
        self.h_shape = input_shape[2]
        self.w_shape = input_shape[3]
        self.ck_shape = input_shape[4]

    def call(self, inputs, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else True
        if training:

            mask = np.random.binomial(1, self.p_seq, [self.batch_shape, self.seq_size])
            mask[:, 0] = 1
            mask = tf.cast(mask, tf.bool)
            if np.random.random() < self.p:
                return inputs, mask

            drop_ch_ind = np.random.choice(np.arange(3), 1, replace=False)
            drop_out = 1 - tf.reduce_sum(tf.one_hot(drop_ch_ind, 3), -2)
            return inputs * drop_out, mask
        else:
            mask = tf.concat([tf.ones([self.batch_shape, 1], dtype=tf.int32), tf.zeros([self.batch_shape, self.seq_size-1], dtype=tf.int32)], -1)
            mask = tf.cast(mask, tf.bool)
            return inputs, mask

    """
    def call(self, inputs, **kwargs):
        training = kwargs['training'] if 'training' in kwargs else True
        if training:
            if np.random.random() < self.p:
                return inputs

            drop_ch_num = int(np.random.choice(np.arange(1, 2 + 1), 1))
            drop_ch_ind = np.random.choice(np.arange(3), drop_ch_num, replace=False)
            drop_out = 1 - tf.reduce_sum(tf.one_hot(drop_ch_ind, 3), -2)
            return inputs * drop_out
        else:
            return inputs"""

    def get_config(self):
        return {'p': self.p, 'p_seq':self.p_seq}