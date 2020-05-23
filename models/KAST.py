import tensorflow as tf
import numpy as np
from ResNet import ResNet
from RKNModel import RKNModel
from Memory import Memory
from Transformation import Transformation

class KAST(tf.keras.Model):
    def __init__(self, coef_memory=0.1, dropout_seq=0.9):
        super(KAST, self).__init__()
        self.kernel = 15
        self.dropout_seq = dropout_seq
        self.transformation = Transformation(trainable=False)
        self.resnet = ResNet()
        self.rkn = RKNModel()
        self.memory = Memory(unit=100, kernel=self.kernel)
        #self.memory = tf.keras.Sequential()
        #self.memory.add(tf.keras.layers.Input(input_shape=((None, None, 256)), batch_input_shape=[4]))
        #self.memory.add(tf.keras.layers.RNN(self.memory_cell, stateful=True))
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
            score = self.rkn((k, tf.reshape(seq_mask, [bs, seq_size, 1])))

        previous_v = v[:, 0]
        self.memory.get_init_state(bs)
        self.memory.call_init((tf.reshape(k[:, 0], [bs, h * w, ck]), tf.reshape(previous_v, [bs, h * w, cv]), tf.reshape(score[:, 0], [bs, h * w])), bs)
        for i in range(seq_size-1):
            with tf.name_scope('Memory'):
                m_kv = self.memory.call((tf.reshape(k[:, i], [bs, h*w, ck]), tf.reshape(previous_v, [bs, h*w, cv]), tf.reshape(score[:, i], [bs, h*w])))
                m_k, m_v = tf.nest.flatten(m_kv) # (bs, m, kernel**2 * k), (bs, m, kernel**2 * v)

            #km_k = tf.concat([tf.reshape(k[:, i], [-1, h*w, ck]), m_k], 1)  # (bs, h*w + m, ck)
            #vm_v = tf.concat([tf.reshape(previous_v, [-1, h*w, cv]), m_v], 1)  # (bs, h*w + m, cv)
            with tf.name_scope('Similarity_matrix'):
                output_v_i = self._get_output_patch(m_k, tf.reshape(k[:, i+1], [-1, h*w, ck]), m_v)  # (bs, nb_patch, h*w+m)
            #with tf.name_scope('Similarity_K'):
            #    similarity_k = self._get_affinity_matrix(tf.reshape(k[:, i], [-1, h*w, ck]), tf.reshape(k[:, i+1], [-1, h*w, ck])) # (bs, h*w, h*w)
            #with tf.name_scope('Similarity_M'):
            #    similarity_m = self._get_affinity_matrix(m_k, tf.reshape(k[:, i+1], [-1, h * w, ck]))  # (bs, h*w, m)


            #reconstruction_k = similarity_k @ tf.reshape(previous_v, [-1, h * w, cv])  # (bs, h*w, v)
            #reconstruction_m = similarity_m @ m_v
            #output_v_i = similarity @ vm_v
            #output_v_i = (1 - self.coef_memory) * reconstruction_k + self.coef_memory * reconstruction_m
            previous_v = tf.where(tf.reshape(seq_mask[:, i+1], [bs, 1, 1, 1]), v[:, i], tf.reshape(output_v_i, [-1, h, w, cv]))
            output_v_i = tf.reshape(output_v_i, [-1, 1, h, w, cv])
            output_v.append(output_v_i)
            ground_truth_i = tf.reshape(v[:, i+1], [-1, 1, h, w, cv])
            ground_truth.append(ground_truth_i)


        #print("output_v len: ", len(output_v))
        #print("output_v[0].shape: ", output_v[0].shape)
        #print("ground_truth len: ", len(ground_truth))
        #print("ground_truth[0].shape: ", ground_truth[0].shape)


        #self.memory.get_initial_state()

        return tf.concat(output_v, 1), tf.concat(ground_truth, 1), i_drop

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
        i_raw, v = tf.nest.flatten(inputs)

        #with tf.name_scope('Transformation'):
        #    i_drop = self.transformation(i_raw, **kwargs)
        with tf.name_scope('ResNet'):
            k = tf.reshape(self.resnet(tf.reshape(i_raw, [-1, H, W, C])),
                           [bs, seq_size, h, w, 256])  # (bs, T, h, w, 256)

        mask = np.random.binomial(1, 0.2, [bs, seq_size])
        mask[:, 0] = 1
        mask = tf.cast(mask, tf.bool)

        mask = tf.reshape(mask, [bs, seq_size, 1])

        with tf.name_scope('Rkn'):
            rkn_k = self.rkn((k, mask))

        return rkn_k, k

    def call_Score(self, inputs, **kwargs):
        # inputs: [(bs, T, H, W, 3), (bs, T, h, w, 3)]
        bs = inputs[1].shape[0]
        seq_size = inputs[1].shape[1]
        H = inputs[0].shape[2]
        W = inputs[0].shape[3]
        C = inputs[0].shape[4]
        h = inputs[1].shape[2]
        w = inputs[1].shape[3]
        cv = inputs[1].shape[4]
        i_raw, v = tf.nest.flatten(inputs)

        # with tf.name_scope('Transformation'):
        #    i_drop = self.transformation(i_raw, **kwargs)
        with tf.name_scope('ResNet'):
            k = tf.reshape(self.resnet(tf.reshape(i_raw, [-1, H, W, C])),
                           [bs, seq_size, h, w, 256])  # (bs, T, h, w, 256)

        mask = np.random.binomial(1, 0.9, [bs, seq_size])
        mask[:, 0] = 1
        mask = tf.cast(mask, tf.bool)

        mask = tf.reshape(mask, [bs, seq_size, 1])

        with tf.name_scope('Rkn'):
            rkn_score = self.rkn((k, mask))

        mask_score = tf.concat([tf.ones([bs, 1, h*w, 1]), tf.zeros([bs, seq_size-1, h*w, 1])], 1)
        k = tf.reshape(k, [bs, seq_size, h*w, 256])
        v = tf.reshape(v, [bs, seq_size, h*w, 3])
        rkn_score = tf.reshape(rkn_score, [bs, seq_size, h*w, 1]) * mask_score

        with tf.name_scope("Memory"):
            mem = self.memory((k, v, rkn_score))
            m_k, m_v, m_u, m_rkn_score = tf.nest.flatten(mem)
            m_u = tf.expand_dims(m_u, -1)

        return m_rkn_score[:, 0], m_u[:, 4]


    def _get_affinity_matrix(self, ref, tar):
        # (bs, h*w + m, k), (bs, h*w, k)
        ref_transpose = tf.transpose(ref, [0, 2, 1])
        inner_product = tar @ ref_transpose
        similarity = tf.nn.softmax(inner_product, -1)
        return similarity  # (bs, h*w, h*w+m)

    def _get_output_patch(self, m_k, k_next, m_v):
        # (bs, m, kernel**2, k), (bs, h*w, k)
        print("m_k.shape: ", m_k.shape)
        print("k_next.shape: ", k_next.shape)
        print("m_v.shape: ", m_v.shape)
        m_k_patch_center = m_k[:, :, (self.kernel**2)//2+1, :]
        print("m_k_patch_center.shape: ", m_k_patch_center.shape)
        ref_transpose = tf.transpose(m_k_patch_center, [0, 2, 1])  # (bs, k, m)
        print("ref_transpose.shape: ", ref_transpose.shape)
        inner_product = k_next @ ref_transpose
        print("inner_product.shape: ", inner_product.shape)
        max_patch = tf.argmax(inner_product, -1)
        out_arr = []
        k_next = tf.unstack(tf.expand_dims(k_next, -2), num=4096, axis=1)
        max_patch = tf.unstack(max_patch, num=4096, axis=1)
        m_k = tf.transpose(m_k, [0, 1, 3, 2])
        m_v = tf.transpose(m_v, [0, 1, 3, 2])
        for i in range(4096):
            m_k_one_patch = tf.gather(m_k, max_patch[i], batch_dims=1, axis=1)
            m_v_one_patch = tf.gather(m_v, max_patch[i], batch_dims=1, axis=1)
            sim = tf.nn.softmax(k_next[i] @ m_k_one_patch)  # (bs, 1, 225)
            out_v = sim @ m_v_one_patch
            out_arr.append(out_v)

        output_i = tf.stack(out_arr, axis=1)

        #print("max_patch.shape: ", max_patch.shape)
        #m_k_one_patch = tf.gather(m_k, max_patch, batch_dims=1, axis=1)  # (bs, hw, kernel**2, 256)
        #print("m_k_one_patch.shape: ", m_k_one_patch.shape)
        #m_v_one_patch = tf.gather(m_v, max_patch, batch_dims=1, axis=1)
        #print("m_v_one_patch.shape: ", m_v_one_patch.shape)
        #inner_product = tf.expand_dims(k_next, -2) @ tf.transpose(m_k_one_patch, [0, 1, 3, 2]) # (bs, hw, 1, 256) @ (bs, hw, 256, kernel**2)  = (bs, hw, 1, kernel**2)
        #print("inner_product.shape: ", inner_product.shape)
        #similarity = tf.nn.softmax(inner_product, -1)
        #print("similarity.shape: ", similarity.shape)
        #output_i = similarity @ m_v_one_patch  # (bs, hw, 1, kernel**2) @ (bs, hw, kernel**2, 3)
        print("output_i.shape: ", output_i.shape)
        return output_i  # (bs, h*w, h*w+m)

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
        output_v, v_j, _ = self.call((inputs, v), training=True)
        #output_v, v_j = self.call_ResNet((inputs, v), training=True)
        #rkn_k, k = self.call_RKN((inputs, v), training=True)
        #rkn_score, m_rkn_score = self.call_Score((inputs, v), training=True)
        abs = tf.math.abs(output_v - v_j)
        loss = tf.reduce_mean(tf.where(abs < 1., 0.5*abs*abs, abs-0.5))
        #loss = -tf.reduce_mean(self.log_normal_pdf(rkn_k, k, tf.math.log(0.001)))
        #loss = tf.reduce_mean(tf.square(rkn_score - m_rkn_score))
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
        output_v, v_j, _ = self.call((inputs, v), training=False)
        #output_v, v_j = self.call_ResNet((inputs, v), training=False)
        #rkn_k, k = self.call_RKN((inputs, v), training=False)
        #rkn_score, m_rkn_score = self.call_Score((inputs, v), training=False)
        return tf.reduce_mean(tf.square(output_v - v_j))

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        #gradients = tape.gradient(loss, self.trainable_variables)
        gradients = tape.gradient(loss, self.resnet.trainable_variables)
        #optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        optimizer.apply_gradients(zip(gradients, self.resnet.trainable_variables))
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
        output_v, v_j, drop_out = self.call((inputs, v), training=training)
        drop_out = tf.reshape(drop_out, [-1, H, W, cv])
        drop_out = tf.image.resize(drop_out, [h, w])
        drop_out = tf.reshape(drop_out, [-1, seq_size, h, w, cv])
        return output_v, v, drop_out
