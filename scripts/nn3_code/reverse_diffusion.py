#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 3, morning
# Erik Spence.
#
#
# This file, reverse_diffusion.py, contains the code which generates a
# new image from the trained model.
#

#######################################################################


import tensorflow as tf
import tensorflow.keras.models as km

import numpy as np

from unet import *
from ddpm import ddpm_schedules


#######################################################################


def my_sample(image_shape = [64, 64, 3], n_sample = 1):

    ## The noise schedule.
    beta1, beta2, T = 1e-4, 0.02, 1000
    schedules = ddpm_schedules(beta1, beta2, T)
    
    ## The size of our generated image.
    this_size = list(image_shape)
    this_size.insert(0, n_sample)

    ## Initial image.
    x_t = tf.random.normal(this_size)

    ## Our trained model.
    model = km.load('model.keras')


    ## Now denoise.
    for i in range(T, -1, -1):
        
        print(i)

        ## Generate the noise for the reparameterization.
        z = tf.random.normal(this_size) if i > 0 else 0

        ## Scale t.
        t = tf.repeat(tf.constant(i / T), n_sample)
        
        ## Predict the noise to be take away from the existing image.
        eps = model.predict([x_t, t], verbose = 0)

        ## Generate the new image.
        x_t = schedules['oneover_sqrta'][i] * \
            (x_t - eps * schedules['mab_over_sqrtmab'][i]) + \
            schedules['sqrt_beta_t'][i] * z

    ## Return the result.
    return np.array(x_t * 255).astype('int16')


#######################################################################
