#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 1
# Erik Spence.
#
# This file, example1.py, contains a routine to generate the data for
# the first example.
#

#######################################################################


"""
example1.py contains a routine for generating the data for the class'
first example.

"""


#######################################################################


import sklearn.datasets as skd
import sklearn.model_selection as skms


#######################################################################


def get_data(n):

    """
    This functon generates n data points generated using
    scikit-learn's make_blobs function.  The data is then split into
    training and testing data sets, and returned.

    Inputs:
    - n: int, the number of points to return.

    Outputs: two arrays of size 0.8 * n, 0.2 * n, randomly generated
    using scikit-learn's make_blobs routine.

    """

    # Generate a synthetid dataset.
    # n: the total number of points to generate, default=100
    # n_features: the number of features for each sample, default=2
    # centers: the number of centers to generate, or the fixed center locations, default=3
    # center_box: the bounding box of each cluster center when cetners are generated at random, default=(-10, 10)
    # cluster_std: the standard deviation of the clusters, default=1
    # shuffle: whether to shuffle the samples, default=True
    # random_state: determines random number generated for dataset creation, default=None
    pos, value = skd.make_blobs(n, centers = 2,
                                center_box = (-3, 3),
                                cluster_std = 1.0)

    # Split the data into training and testing data sets, and return.
    # arrays: one or more datasets to be split
    return skms.train_test_split(pos, value,
                                 test_size = 0.2)


#######################################################################
