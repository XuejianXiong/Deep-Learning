import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# Download and import the MIT Introduction to Deep Learning package
import mitdeeplearning as mdl


class MyDenseLayer(tf.keras.layers.Layer):
    def __int__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__int__()

        # Initialize weights and bias
        self.W = self.add_weight([input_dim, output_dim])
        self.B = self.add_weight([1, output_dim])

        def call(self, inputs):
            # Forward propagate the inputs
            z = tf.matmual(inputs, self.W) + self.b

            # Feed through a non-linear activation
            output = tf.math.sigmoid(z)

            return output


def main():

    layer = tf.keras.layers.Dense(units=2)
    print(layer)

    n = 5
    model = tf.keras.Sequential([tf.keras.layers.Dense(n), 
                                 tf.keras.layers.Dense(2)])
    print(model)


if __name__ == "__main__":
    main()      
