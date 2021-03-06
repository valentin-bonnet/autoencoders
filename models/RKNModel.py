import tensorflow as tf
from RKNLayer import RKNLayer

class RKNModel(tf.keras.Model):
    def __init__(self, score_output=1, layer_channel_1=64, layer_channel_2=128, input_shape=64, k=15, b=7, n=128, alpha_unit=128):
        super(RKNModel, self).__init__()
        ## ENCODER

        self.N = n
        self.score = score_output

        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_1, kernel_size=7, strides=1, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_2, kernel_size=5, strides=1, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(n))

        ## RKN

        cell = RKNLayer(n//2, k, b, alpha_unit)
        self.rkn_layer = tf.keras.layers.RNN(cell, return_sequences=True)

        ## DECODER

        size_decoded_frame = int(input_shape // 4)
        size_decoded_layers = int(layer_channel_2 // 2)
        dense_output = size_decoded_frame**2 * size_decoded_layers

        """
        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.Dense(dense_output))
        self.generative_net.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=5, strides=1, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=7, strides=1, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding='same'))
        """


        self.score_net = tf.keras.Sequential()
        self.score_net.add(tf.keras.layers.Dense(dense_output))
        self.score_net.add(
            tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        self.score_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=5, strides=1, padding='same'))
        self.score_net.add(tf.keras.layers.ReLU())
        self.score_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.score_net.add(tf.keras.layers.ReLU())
        self.score_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=7, strides=1, padding='same'))
        self.score_net.add(tf.keras.layers.ReLU())
        self.score_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.score_net.add(tf.keras.layers.ReLU())
        self.score_net.add(
            tf.keras.layers.Conv2DTranspose(filters=score_output, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu))


    def call(self, inputs):
        # inputs : (bs, T, H, W, K)
        k, mask = tf.nest.flatten(inputs)
        bs = k.shape[0]
        seq_size = k.shape[1]
        h = k.shape[2]
        w = k.shape[3]
        kv = k.shape[4]
        inputs = tf.reshape(k, [-1, h, w, kv])
        encoded = self.inference_net(inputs)
        encoded = tf.reshape(encoded, [bs, seq_size, self.N])
        state = self.rkn_layer((encoded, mask))
        state = tf.reshape(state, [-1, self.N])
        #output = self.generative_net(state)
        score = self.score_net(state)
        output = tf.reshape(score, [bs, seq_size, h, w])
        return output
