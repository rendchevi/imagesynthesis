import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.initializers import RandomNormal

tf.keras.backend.set_floatx('float32')

def DCGAN_32(D_activation = 'sigmoid', latent_size = 100, kernel_size = 4, kernel_init = RandomNormal(stddev = 0.02), alpha = 0.2, rate = 0.5):

    def net_D():

        # Input Layer --> [32, 32, 1]
        I = Input(shape = [32, 32, 1])

        # Layer 1 --> [16, 16, 64]
        x = layers.Conv2D(filters = 32, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(I)
        x = layers.LeakyReLU(alpha = alpha)(x)

        # Layer 2 --> [8, 8, 64]
        x = layers.Conv2D(filters = 16, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)

        # Layer 3 --> [4, 4, 64]
        x = layers.Conv2D(filters = 8, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)

        # Output Layer -> 1
        x = layers.Flatten()(x)
        x = layers.Dense(units = 1, activation = None)(x)
        O = layers.Activation(D_activation)(x)

        # Define the model
        net = Model(inputs = I, outputs = O, name = 'net_D')

        return net

    def net_G():

        # Input Layer -> [1, 1, latent_size]
        I = Input(shape = [1, 1, latent_size])

        # Layer 1 -> [2, 2, 64]
        x = layers.Conv2DTranspose(filters = 128, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(I)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)
        x = layers.Dropout(rate = rate)(x)

        # Layer 2 -> [4, 4, 64]
        x = layers.Conv2DTranspose(filters = 128, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)
        x = layers.Dropout(rate = rate)(x)

        # Layer 3 -> [8, 8, 64]
        x = layers.Conv2DTranspose(filters = 128, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)
        x = layers.Dropout(rate = rate)(x)

        # Layer 4 -> [16, 16, 64]
        x = layers.Conv2DTranspose(filters = 128, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)
        x = layers.Dropout(rate = rate)(x)

        # Layer 5 -> [32, 32, 64]
        x = layers.Conv2DTranspose(filters = 128, kernel_size = kernel_size, strides = 2, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha = alpha)(x)

        # Output Layer -> [32, 32, 1]
        x = layers.Conv2DTranspose(filters = 1, kernel_size = kernel_size, strides = 1, padding = 'same', kernel_initializer = kernel_init, use_bias = False)(x)
        O = layers.Activation('tanh')(x)        

        # Define the model
        net = Model(inputs = I, outputs = O, name = 'net_G')

        return net

    return net_D(), net_G()