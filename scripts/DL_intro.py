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

            ## Feed through a non-linear activation
            # Sigmoid Function
            output = tf.math.sigmoid(z)
            # Hyperbolic Tangent
            #output = tf.math.tanh(z)
            # Rectified Linear Unit (ReLU)
            #output = tf.nn.relu(z)

            return output


def main():

    layer = tf.keras.layers.Dense(units=2)
    print(layer)

    n1 = 5
    n2 = 3
    model = tf.keras.Sequential([
        # 1st layer with n nodes
        tf.keras.layers.Dense(n1),
        # 2nd layer with n nodes
        tf.keras.layers.Dense(n2),
        # ouput layer with 2 nodes
        tf.keras.layers.Dense(2)
    ])
    print(model)


if __name__ == "__main__":
    main()      
