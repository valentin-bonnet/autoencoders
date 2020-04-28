import tensorflow as tf
from ResNet import ResNet
from RKNModel import RKNModel
from Memory import Memory

class KAST(tf.keras.Model):
    def __init__(self, coef_memory=0.2):
        super(KAST, self).__init__()
        self.resnet = ResNet()
        self.rkn = RKNModel()
        self.memory = Memory()
        self.coef_memory = coef_memory
        self.description = 'KAST'

    def call(self, inputs):
        # inputs: [(bs, T, H, W, 3), (bs, T, h, w, 3)]
        seq_size = inputs[0].shape[1]
        output_v = [seq_size-1]
        ground_truth = [seq_size-1]
        i, v = tf.nest.flatten(inputs)

        with tf.name_scope('ResNet'):
            k = self.resnet(i) # (bs, T, h, w, 256)

        h = k.shape[2]
        w = k.shape[3]
        c = k.shape[4]
        for i in range(seq_size-1):
            k_i = k[:, i]
            k_j = k[:, i+1]
            v_i = v[:, i] # (bs, h, w, v)
            v_j = v[:, i+1]
            with tf.name_scope('Rkn'):
                attention = self.rkn(k_i)
            with tf.name_scope('Memory'):
                m_kv = self.memory([attention, k_i, v_i])
                m_k, m_v = tf.nest.flatten(m_kv)

            with tf.name_scope('Similarity_K'):
                similarity_k = self._get_affinity_matrix(tf.reshape(k_i, [-1, h*w, c]), tf.reshape(k_j, [-1, h*w, c])) # (bs, h*w, h*w)
            with tf.name_scope('Similarity_M'):
                similarity_m = self._get_affinity_matrix(m_k, tf.reshape(k_j, [-1, h*w, c])) # (bs, h*w, m)

            reconstruction_k = similarity_k @ tf.reshape(v_i, [-1, h*w, v]) # (bs, h*w, v)
            reconstruction_m = similarity_m @ m_v
            output_v[i] = (1 - self.coef_memory) * reconstruction_k + self.coef_memory * reconstruction_m
            ground_truth[i] = v_j
        return output_v

    def _get_affinity_matrix(self, ref, tar):
        # (bs, h*w or m, k), (bs, h*w, k)
        ref_transpose = tf.transpose(ref, [0, 2, 1])
        inner_product = tar @ ref_transpose
        similarity = tf.nn.softmax(inner_product, -1)
        return similarity

    def set_coef_memory(self, coef_memory):
        if coef_memory < 0:
            coef_memory = 0
        elif coef_memory > 1:
            coef_memory = 1
        return coef_memory

    def compute_loss(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        v = tf.image.resize(inputs, [h//4, w//4])
        output_v, v_j = self.call((inputs, v))
        abs = tf.math.abs(output_v - v_j)
        loss = tf.reduce_mean(tf.where(abs < 1, 0.5*abs*abs, abs-0.5))
        return loss

    def compute_accuracy(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        v = tf.image.resize(inputs, [h // 4, w // 4])
        output_v, v_j = self.call((inputs, v))
        return tf.reduce_mean(tf.square(output_v - v_j))

    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def reconstruct(self, inputs):
        output_v, _ = self.call(inputs)
        return output_v