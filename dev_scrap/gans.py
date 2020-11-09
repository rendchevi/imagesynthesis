import numpy as np
from glob import glob
from time import time

import tensorflow as tf

from ..utils.data_utils import *
from ..utils.misc_utils import *

class GAN:

    def __init__(self, model_name = 'unconditional_imagesynth'):
        self.model_name = model_name
        self.losses_D = []
        self.losses_G = []

        self.DEBUG_time_tracker = []

    def set_network(self, net):
        self.net_D = net[0]
        self.net_G = net[1]
        self.latent_size = self.net_G.input_shape[-1]

    def set_optimizers(self, optim_D, optim_G):
        self.optim_D = optim_D
        self.optim_G = optim_G

    def set_objective_function(self, objective_function):
        self.objective_function = objective_function

    def set_train_files(self, train_files_path):
        self.train_files_path = np.array(glob(train_files_path))
        
    def fit(self, iterations, batch_size = 16, num_train_D = 1, num_train_G = 1):

        for iteration in range(1, iterations + 1):

            # (DEBUG) Time tracker
            start_iter_time = time()

            # Optimizing D
            for j in range(0, num_train_D):
                # Sample a batch of real data
                sample_idxs = np.random.randint(0, len(self.train_files_path), batch_size)
                real_files = self.train_files_path[sample_idxs].copy()
                real_data = process_images(real_files, self.net_D.input_shape[1:3])
                # Sample a batch of fake data
                sample_noise = tf.random.normal([batch_size] + list(self.net_G.input_shape[1:]))
                fake_data = self.net_G(sample_noise)
                # Optimize D
                loss_D = self.objective_function.optimize_D(real_data, fake_data, self.net_D, self.net_G, self.optim_D).numpy()
                
                # Record the loss
                self.losses_D.append(loss_D)
                
            # Optimizing G
            for k in range(0, num_train_G):
                # Sample noise from latent space
                sample_noise = tf.random.normal([batch_size] + list(self.net_G.input_shape[1:]))
                # Optimize G
                loss_G = self.objective_function.optimize_G(sample_noise, self.net_D, self.net_G, self.optim_G).numpy()

                # Record the loss
                self.losses_G.append(loss_G)
            
            # Report the loss
            iter_duration = time() - start_iter_time
            self.DEBUG_time_tracker.append(iter_duration)
            report_loss_simple(iteration, iterations, iter_duration, loss_D, loss_G)
        
    def synthesize_random(self, num_images = 1):
        # Synthesize random image
        sample_noise = tf.random.normal([num_images] + list(self.net_G.input_shape[1:]))
        images = self.net_G(sample_noise, training = False)
        # De-process images
        images = deprocess_images(images)
        
        return images

    def save_model(self):
        pass

    def load_model(self):
        pass