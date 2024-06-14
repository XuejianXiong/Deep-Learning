#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 3, morning
# Erik Spence.
#
#
# This file, diffusion_model.py, contains the code which trains the
# diffusion network.
#

#######################################################################

## Heavily inspired by https://github.com/cloneofsimo/minDiffusion.git

#######################################################################


from unet import NaiveUnet
from ddpm import ddpm_schedules

import tensorflow as tf
import tensorflow.keras.models as km

import os
import os.path
import numpy as np


#######################################################################


gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

    
#######################################################################

## Some parameters
n_epochs = 300
n_feat = 512 
batch_size = 64
print('batch size: ', batch_size, ' n_feat: ', n_feat)


## Get the data.
print('Loading data.', flush = True)
#project = os.environ['PROJECT']
#data_path = project + '/data/dogs-cats'
data_path = '.'

image_shape = (64, 64, 3)

## Get the data, and normalize.
data = np.load(data_path + '/64x64_cats.npy')
data = (data / 255).astype('float32')

## Convert the data to a Tensorflow dataset.
train_dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)


## If a model exists, reload it.  Otherwise, build a fresh one.
if os.path.isfile('model.keras'):

    print('Reloading an existing model.', flush = True)
    model = km.load_model('model.keras')

else:

    print('Building a fresh model.', flush = True)
    model = NaiveUnet(image_shape, n_feat)
    model.compile(loss = 'mse', metrics = ['mse'],
                  optimizer = 'adam')

print(model.summary())

    
## Now create the noise schedule.
beta1, beta2 = 1e-4, 0.02
T = 1000

## Build the noise schedules.
schedules = ddpm_schedules(beta1, beta2, T)


print('Starting training.', flush = True)
for i in range(n_epochs):

    loss = 0
    
    for image_batch in train_dataset:

        this_batch_size = image_batch.shape[0]
        this_size = list(image_shape)
        this_size.insert(0, this_batch_size)
            
        ## Create some noise
        eps = tf.random.normal(this_size)

        ## Randomly pick some integers
        t_index = tf.random.uniform([this_batch_size],
                                    minval = 1,
                                    maxval = T + 1,
                                    dtype = tf.int32)
        
        ## slice those indexed values out of the schedule
        this_sqrtab = tf.gather(schedules['sqrtab'], t_index)
        this_sqrtmab = tf.gather(schedules['sqrtmab'], t_index)

        ## to make each image be multiplied by a single noise
        ## value we need to reshape the noise values into this
        ## form
        this_sqrtab = tf.reshape(this_sqrtab,
                                 (this_batch_size, 1, 1, 1))
        this_sqrtmab = tf.reshape(this_sqrtmab,
                                  (this_batch_size, 1, 1, 1))


        ## Build x_t.
        x_t = this_sqrtab * image_batch + this_sqrtmab * eps

        ## Scale the time.
        t_input = t_index / T
        
        ## The loss is calculated between the noise, eps, and the
        ## input image, and time value
        loss += model.train_on_batch([x_t, t_input], eps)[0]

    print('Epoch :' + str(i) + ' loss :' + str(loss), flush = True)
        
model.save('model.keras')
