#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 2
# Erik Spence
#
# This file, model1_v2.py, contains a routine to generate the model for
# the first Keras example, using Keras' functional syntax.
#

#######################################################################


"""
model1_v2.py contains a routine which generates the model for the class'
first Keras example, using Keras' functional syntax.

"""


#######################################################################


import tensorflow.keras.models as km
import tensorflow.keras.layers as kl


#######################################################################


def get_model(numnodes, input_size = 784, output_size = 10):

    """
    This function returns a simple Keras model, consisting of a
    re-implementation of the second_network.py neural network, with
    numnodes in the hidden layer, using Keras' functional syntax.

    Inputs:
    - numnodes: int, the number of nodes in the hidden layer.

    - intput_size: int, the size of the input data, default = 784.

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    input_image = kl.Input(shape = input_size, name = 'input')

    # Add a hidden fully-connected layer.
    x = kl.Dense(numnodes, activation = 'sigmoid',
                 name = 'hidden')(input_image)

    # Add the output layer.
    x = kl.Dense(output_size, activation = 'sigmoid',
                 name = 'output')(x)

    model = km.Model(inputs = input_image, outputs = x)

    # Return the model.
    return model


#######################################################################
