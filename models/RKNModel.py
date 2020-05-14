import tensorflow as tf
from RKNLayer import RKNLayer

class RKNModel(tf.keras.Model):
    def __init__(self, attention_output=3, layer_channel_1=64, layer_channel_2=128, input_shape=64, k=15, b=7, n=128, alpha_unit=128):
        super(RKNModel, self).__init__()
        ## ENCODER

        self.N = n
        self.attention = attention_output

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
        self.attention_net = tf.keras.Sequential()
        self.attention_net.add(tf.keras.layers.Dense(dense_output))
        self.attention_net.add(
            tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        self.attention_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=5, strides=1, padding='same'))
        self.attention_net.add(tf.keras.layers.ReLU())
        self.attention_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.attention_net.add(tf.keras.layers.ReLU())
        self.attention_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=7, strides=1, padding='same'))
        self.attention_net.add(tf.keras.layers.ReLU())
        self.attention_net.add(
            tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.attention_net.add(tf.keras.layers.ReLU())
        self.attention_net.add(
            tf.keras.layers.Conv2DTranspose(filters=attention_output, kernel_size=3, strides=1, padding='same'))
            """

    def call(self, inputs):
        # inputs : (bs, T, H, W, K)
        bs = inputs.shape[0]
        seq_size = inputs.shape[1]
        h = inputs.shape[2]
        w = inputs.shape[3]
        k = inputs.shape[4]
        inputs = tf.reshape(inputs, [-1, h, w, k])
        encoded = self.inference_net(inputs)
        encoded = tf.reshape(encoded, [bs, seq_size, self.N])
        state = self.rkn_layer(encoded)
        state = tf.reshape(state, [-1, self.N])
        output = self.generative_net(state)
        output = tf.reshape(output, [bs, seq_size, h, w, 256])
        return output
