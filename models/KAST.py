import tensorflow as tf
import numpy as np
from ResNet import ResNet
from RKNModel import RKNModel
from Memory import Memory
from Transformation import Transformation

class KAST(tf.keras.Model):
    def __init__(self, coef_memory=0.1, dropout_seq=0.6):
        super(KAST, self).__init__()
        self.dropout_seq = dropout_seq
        self.transformation = Transformation(trainable=False)
        self.resnet = ResNet()
        self.rkn = RKNModel()
        self.memory = Memory()
        self.coef_memory = coef_memory
        self.description = 'KAST'

    def call(self, inputs, **kwargs):
        # inputs: [(bs, T, H, W, 3), (bs, T, h, w, 3)]
        bs = inputs[1].shape[0]
        seq_size = inputs[1].shape[1]
        H = inputs[0].shape[2]
        W = inputs[0].shape[3]
        C = inputs[0].shape[4]
        h = inputs[1].shape[2]
        w = inputs[1].shape[3]
        cv = inputs[1].shape[4]
        output_v = []
        ground_truth = []
        i_raw, v = tf.nest.flatten(inputs)
        #print("i.shape: ", i.shape)
        #print("v.shape: ", v.shape)

        with tf.name_scope('Transformation'):
            i_drop, seq_mask = self.transformation(i_raw, **kwargs)
        with tf.name_scope('ResNet'):
            k = tf.reshape(self.resnet(tf.reshape(i_drop, [-1, H, W, C])), [bs, seq_size, h, w, 256]) # (bs, T, h, w, 256)

        ck = k.shape[4]

        with tf.name_scope('Rkn'):
            attention = self.rkn(k)

        previous_v = v[:, 0]

        for i in range(seq_size-1):
            with tf.name_scope('Memory'):
                m_kv = self.memory((attention[:, i], k[:, i], previous_v))
                m_k, m_v = tf.nest.flatten(m_kv)
            with tf.name_scope('Similarity_K'):
                similarity_k = self._get_affinity_matrix(tf.reshape(k[:, i], [-1, h*w, ck]), tf.reshape(k[:, i+1], [-1, h*w, ck])) # (bs, h*w, h*w)
            with tf.name_scope('Similarity_M'):
                similarity_m = self._get_affinity_matrix(m_k, tf.reshape(k[:, i+1], [-1, h * w, ck]))  # (bs, h*w, m)


            reconstruction_k = similarity_k @ tf.reshape(previous_v, [-1, h * w, cv])  # (bs, h*w, v)
            reconstruction_m = similarity_m @ m_v
            output_v_i = (1 - self.coef_memory) * reconstruction_k + self.coef_memory * reconstruction_m
            previous_v = tf.where(tf.reshape(seq_mask[:, i], [bs, 1, 1, 1]), v[:, i], tf.reshape(output_v_i, [-1, h, w, cv]))
            output_v_i = tf.reshape(output_v_i, [-1, 1, h, w, cv])
            output_v.append(output_v_i)
            ground_truth_i = tf.reshape(v[:, i+1], [-1, 1, h, w, cv])
            ground_truth.append(ground_truth_i)


        #print("output_v len: ", len(output_v))
        #print("output_v[0].shape: ", output_v[0].shape)
        #print("ground_truth len: ", len(ground_truth))
        #print("ground_truth[0].shape: ", ground_truth[0].shape)

        dict_view = {
            'input_dropout': i_drop,
            'attention': attention,
        }

        self.memory.get_initial_state()

        return tf.concat(output_v, 1), tf.concat(ground_truth, 1), dict_view

    def call_ResNet(self, inputs, **kwargs):
        # inputs: [(bs, T, H, W, 3), (bs, T, h, w, 3)]
        bs = inputs[1].shape[0]
        seq_size = inputs[1].shape[1]
        H = inputs[0].shape[2]
        W = inputs[0].shape[3]
        C = inputs[0].shape[4]
        h = inputs[1].shape[2]
        w = inputs[1].shape[3]
        cv = inputs[1].shape[4]
        output_v = []
        ground_truth = []
        i_raw, v = tf.nest.flatten(inputs)
        # print("i.shape: ", i.shape)
        # print("v.shape: ", v.shape)

        with tf.name_scope('Transformation'):
            i_drop = self.transformation(i_raw, **kwargs)
        with tf.name_scope('ResNet'):
            k = tf.reshape(self.resnet(tf.reshape(i_drop, [-1, H, W, C])),
                           [bs, seq_size, h, w, 256])  # (bs, T, h, w, 256)

        ck = k.shape[4]

        with tf.name_scope('Similarity_K'):
            similarity_k = self._get_affinity_matrix(tf.reshape(k[:, 0], [-1, h * w, ck]),
                                                     tf.reshape(k[:, 1], [-1, h * w, ck]))  # (bs, h*w, h*w)

        reconstruction_k = tf.reshape(similarity_k @ tf.reshape(v[:, 0], [-1, h * w, cv]), [-1, h, w, cv]) # (bs, h*w, v)
        ground_truth = v[:, 1]
        return reconstruction_k, ground_truth

    def call_RKN(self, inputs, **kwargs):
        # inputs: [(bs, T, H, W, 3), (bs, T, h, w, 3)]
        bs = inputs[1].shape[0]
        seq_size = inputs[1].shape[1]
        H = inputs[0].shape[2]
        W = inputs[0].shape[3]
        C = inputs[0].shape[4]
        h = inputs[1].shape[2]
        w = inputs[1].shape[3]
        cv = inputs[1].shape[4]
        output_v = []
        ground_truth = []
        i_raw, v = tf.nest.flatten(inputs)
        # print("i.shape: ", i.shape)
        # print("v.shape: ", v.shape)

        #with tf.name_scope('Transformation'):
        #    i_drop = self.transformation(i_raw, **kwargs)
        with tf.name_scope('ResNet'):
            k = tf.reshape(self.resnet(tf.reshape(i_raw, [-1, H, W, C])),
                           [bs, seq_size, h, w, 256])  # (bs, T, h, w, 256)

        mask = np.random.binomial(1, 0.9, [bs, seq_size])
        mask[:, 0] = 1

        mask = tf.cast(tf.reshape(mask, [bs, seq_size, 1, 1, 1]), tf.float32)

        k = k*mask

        with tf.name_scope('Rkn'):
            rkn_mean, rkn_std = self.rkn(k)

        return rkn_mean, rkn_std, k


    def _get_affinity_matrix(self, ref, tar):
        # (bs, h*w or m, k), (bs, h*w, k)
        ref_transpose = tf.transpose(ref, [0, 2, 1])
        inner_product = tar @ ref_transpose
        similarity = tf.nn.softmax(inner_product, -1)
        return similarity

    def set_coef_memory(self, coef_memory):
        if coef_memory < 0:
            self.coef_memory = 0
        elif coef_memory > 1:
            self.coef_memory = 1

    def set_dropout_seq(self, dropout_seq):
        if dropout_seq < 0:
            self.dropout_seq = 0
        elif dropout_seq > 1:
            self.dropout_seq = 1
        return dropout_seq

    def log_normal_pdf(self, sample, mean, logvar):
      log2pi = tf.math.log(2. * np.pi)
      return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

    def compute_loss(self, inputs):
        seq_size = inputs.shape[1]
        H = inputs.shape[2]
        W = inputs.shape[3]
        cv = inputs.shape[4]
        h = H//4
        w = W//4
        v = tf.reshape(inputs, [-1, H, W, cv])
        v = tf.image.resize(v, [h, w])
        v = tf.reshape(v, [-1, seq_size, h, w, cv])
        #output_v, v_j, _ = self.call((inputs, v), training=True)
        #output_v, v_j = self.call_ResNet((inputs, v), training=True)
        mean, std, k = self.call_RKN((inputs, v), training=True)
        #abs = tf.math.abs(output_v - v_j)
        #loss = tf.reduce_mean(tf.where(abs < 1, 0.5*abs*abs, abs-0.5))
        loss = tf.reduce_mean(self.log_normal_pdf(k, mean, std))
        print(loss)
        return loss

    def compute_accuracy(self, inputs):
        seq_size = inputs.shape[1]
        H = inputs.shape[2]
        W = inputs.shape[3]
        cv = inputs.shape[4]
        h = H // 4
        w = W // 4
        v = tf.reshape(inputs, [-1, H, W, cv])
        v = tf.image.resize(v, [h, w])
        v = tf.reshape(v, [-1, seq_size, h, w, cv])
        #output_v, v_j, _ = self.call((inputs, v), training=False)
        #output_v, v_j = self.call_ResNet((inputs, v), training=False)
        mean, _, k = self.call_RKN((inputs, v), training=False)
        return tf.reduce_mean(tf.square(mean - k))

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        #gradients = tape.gradient(loss, self.trainable_variables)
        gradients = tape.gradient(loss, self.rkn.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        optimizer.apply_gradients(zip(gradients, self.rkn.trainable_variables))
        return loss

    def reconstruct_ResNet(self, inputs, training=True):
        seq_size = inputs.shape[1]
        H = inputs.shape[2]
        W = inputs.shape[3]
        cv = inputs.shape[4]
        h = H // 4
        w = W // 4
        v = tf.reshape(inputs, [-1, H, W, cv])
        v = tf.image.resize(v, [h, w])
        v = tf.reshape(v, [-1, seq_size, h, w, cv])
        output_v, v_j = self.call_ResNet((inputs, v), training=training)
        return output_v, v_j

    def reconstruct(self, inputs, training=True):
        seq_size = inputs.shape[1]
        H = inputs.shape[2]
        W = inputs.shape[3]
        cv = inputs.shape[4]
        h = H // 4
        w = W // 4
        v = tf.reshape(inputs, [-1, H, W, cv])
        v = tf.image.resize(v, [h, w])
        v = tf.reshape(v, [-1, seq_size, h, w, cv])
        output_v, v_j, dict_view = self.call((inputs, v), training=training)
        drop_out = dict_view['input_dropout']
        drop_out = tf.reshape(drop_out, [-1, H, W, cv])
        drop_out = tf.image.resize(drop_out, [h, w])
        drop_out = tf.reshape(drop_out, [-1, seq_size, h, w, cv])
        dict_view['input_dropout'] = drop_out
        return output_v, v, dict_view
