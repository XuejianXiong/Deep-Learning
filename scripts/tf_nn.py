import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Download and import the MIT Introduction to Deep Learning package
import mitdeeplearning as mdl


def add_layer(model, n_nodes):
      # Define one Dnse layer
      dense_layer = Dense(units=n_nodes)

      # Add the Dense layer to the model
      model.add(dense_layer)   

      return model  


def build_model(node_list):
      # First define the model
      model = Sequential()

      # Add layers to the model
      for n1 in node_list:
            model = add_layer(model, n1)

      return model
     


def main():
      ### Defining a neural network using the Sequential API ###

      node_list = [10, 10, 3]

      model = build_model(node_list)

      # Test model with example input
      x_input = tf.constant([[1, 2.]], shape=(1,2))
      print(x_input)

      # Feed input into the model and predict the output
      model_output = model(x_input)
      print(model_output)



if __name__ == "__main__":
    main()      
