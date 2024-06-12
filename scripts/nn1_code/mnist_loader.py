#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 1
# Erik Spence.
#
# This file, mnist_loader.py, contains the code needed to load the
# MNIST dataset.  The code borrows heavily from
# http://neuralnetworksanddeeplearning.com.
#

#######################################################################


"""
mnist_loader contains the code needed to load the MNIST dataset,
and convert it to 1D.  The code has been heavily borrowed from
http://neuralnetworksanddeeplearning.com.

"""


#######################################################################


import pickle as cPickle

import gzip
import numpy as np
import sys


#######################################################################


def get_data(filename):

    # Open the file.
    f = gzip.open(filename, 'rb')

    # Load the data, but check to see if it's Python 3.
    if (sys.version_info.major == 3):
        training_data, validation_data, test_data = cPickle.load(f, encoding = 'latin1')
    else:
        training_data, validation_data, test_data = cPickle.load(f)
        
    # Close the file.
    f.close()

    return training_data, validation_data, test_data


#######################################################################


def load_mnist_1D_small(filename = '../data/mnist.pkl.gz'):

    """
    Returns a small version of the MNIST data as a tuple containing
    the training data, the validation data, and the test data.

    Inputs:
    
    - filename: string, name of the file containing the data.

    Outputs:
    
    - tuple, containing the training and test data.  These
      take the form:

        - training_data: tuple, consisting of:
   
            - 2D array of floats of shape (500, 784), containing the
              pixel values for each image.  

            - integer vector of length 500, containing the value of
              the number in the image.
       
        - test_data: same as training_data, except with length
          100

    """

    # Load the data.
    training_data, validation_data, test_data = get_data(filename)

    # Return the values.
    return training_data[0][0:500, :], test_data[0][0:100, :], \
        training_data[1][0:500], test_data[1][0:100]


