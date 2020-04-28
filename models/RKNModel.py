import tensorflow as tf
from RKNLayer import RKNLayer

class RKNModel(tf.keras.Model):
    def __init__(self, attention_output=3, layer_channel_1=64, layer_channel_2=128, input_shape=16, k=15, b=15, n=128, alpha_unit=64):
        super(RKNModel, self).__init__()
        ## ENCODER

        self.inference_net = tf.keras.Sequential()
        self.inference_net.add(tf.keras.layers.Input(shape=(input_shape, input_shape, 256)))
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_1, kernel_size=3, strides=1, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())
        self.inference_net.add(tf.keras.layers.Conv2D(filters=layer_channel_2, kernel_size=3, strides=1, padding='same'))
        self.inference_net.add(tf.keras.layers.ReLU())

        self.inference_net.add(tf.keras.layers.Flatten())
        self.inference_net.add(tf.keras.layers.Dense(n))

        ## RKN

        cell = RKNLayer(n//2, k, b, alpha_unit)
        self.rkn = tf.keras.layers.RNN(cell)

        ## DECODER

        size_decoded_frame = int(input_shape // 4)
        size_decoded_layers = int(layer_channel_2 // 2)
        dense_output = size_decoded_frame**2 * size_decoded_layers

        self.generative_net = tf.keras.Sequential()
        self.generative_net.add(tf.keras.layers.InputLayer(input_shape=(n)))
        self.generative_net.add(tf.keras.layers.Dense(dense_output))
        self.generative_net.add(tf.keras.layers.Reshape(target_shape=(size_decoded_frame, size_decoded_frame, size_decoded_layers)))
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=3, strides=1, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_2, kernel_size=3, strides=2, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=3, strides=1, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=layer_channel_1, kernel_size=3, strides=2, padding='same'))
        self.generative_net.add(tf.keras.layers.ReLU())
        self.generative_net.add(tf.keras.layers.Conv2DTranspose(filters=attention_output, kernel_size=3, strides=1, activation='sigmoid', padding='same'))



    def call(self, inputs):
        # inputs : (bs, T, H, W, K)
        print("RKNModel input shape: ", inputs.shape)
        encoded = self.inference_net(inputs)
        print("RKNModel encoded shape: ", encoded.shape)
        state = self.rkn(encoded)
        output = self.generative_net(state)
        return output
