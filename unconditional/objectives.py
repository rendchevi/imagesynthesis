import tensorflow as tf

from ..utils.net_utils import *

class NonSaturatingGame:

    def __init__(self):
        self.objective_name = 'non_saturating_game'

    @tf.function()
    def optimize_D(self, real_data, fake_data, net_D, net_G, optim_D):

        # Keeping tracks of gradients from below computations
        with tf.GradientTape() as D_tape:

            # Get D output fed with real data
            D_output_real = net_D(real_data)

            # Get discriminator output fed with fake data
            D_output_fake = net_D(fake_data)

            # Compute D loss
            # --> Maximizing log(D(x)) + log(1 - D(G(x))) 
            loss_D = -tf.reduce_mean(tf.math.log(D_output_real) + tf.math.log(1.0 - D_output_fake))

        # Compute gradients using the tape
        grads_D = D_tape.gradient(loss_D, net_D.trainable_variables)

        # Optimize the model (updating the weights)
        optim_D.apply_gradients(zip(grads_D, net_D.trainable_variables))

        return loss_D
        
    @tf.function()
    def optimize_G(self, sample_noise, net_D, net_G, optim_G):

        # Keeping tracks of gradients from below computations
        with tf.GradientTape() as G_tape:

            # Get D output fed with fake data
            fake_data = net_G(sample_noise)
            D_output_fake = net_D(fake_data)

            # Compute G loss
            # --> Maximizing log(D(G(z))) where z is latent noise
            loss_G = -tf.reduce_mean(tf.math.log(D_output_fake))

        # Compute gradients using the tape
        grads_G = G_tape.gradient(loss_G, net_G.trainable_variables)

        # Optimize the model (updating the weights)
        optim_G.apply_gradients(zip(grads_G, net_G.trainable_variables))

        return loss_G

class Wasserstein:

    def __init__(self, regularization_method = 'weight_clipping', clip_value = (-0.01, 0.01)):
        self.objective_name = 'wasserstein'

        self.regularization_method = regularization_method
        self.clip_value = clip_value

    #@tf.function()
    def optimize_D(self, real_data, fake_data, net_D, net_G, optim_D):

        # Keeping tracks of gradients from below computations
        with tf.GradientTape() as D_tape:

            # Get D output fed with real data
            D_output_real = net_D(real_data)

            # Get discriminator output fed with fake data
            D_output_fake = net_D(fake_data)

            # Compute D loss
            # --> Minimizing D(x)) - D(G(z)) 
            #loss_D = tf.reduce_mean(D_output_real) - tf.reduce_mean(D_output_fake)
            loss_D = -(tf.reduce_mean(D_output_real) - tf.reduce_mean(D_output_fake))

        # Compute gradients using the tape
        grads_D = D_tape.gradient(loss_D, net_D.trainable_variables)

        # Optimize the model (updating the weights)
        optim_D.apply_gradients(zip(grads_D, net_D.trainable_variables))

        # Perform regularization
        weight_clipping(net_D, clip_value = self.clip_value)

        return loss_D
        
    #@tf.function()
    def optimize_G(self, sample_noise, net_D, net_G, optim_G):

        # Keeping tracks of gradients from below computations
        with tf.GradientTape() as G_tape:

            # Get D output fed with fake data
            fake_data = net_G(sample_noise)
            D_output_fake = net_D(fake_data)

            # Compute G loss
            # --> Maximizing log(D(G(z))) where z is latent noise
            loss_G = -tf.reduce_mean(D_output_fake)

        # Compute gradients using the tape
        grads_G = G_tape.gradient(loss_G, net_G.trainable_variables)

        # Optimize the model (updating the weights)
        optim_G.apply_gradients(zip(grads_G, net_G.trainable_variables))

        return loss_G