import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

import tensorflow as tf

# Download and import the MIT Introduction to Deep Learning package
import mitdeeplearning as mdl

def get_nd_tensor(
      shape: Tuple[int, ...], 
      init_type: str = "random"
) -> tf.Tensor:
      
      if init_type == "random":
            return tf.random.normal(shape)      
      elif init_type == "zeros":
            return tf.zeros(shape)
      elif init_type == "ones":
            return tf.ones(shape)
      else:
            raise ValueError("init_type must be 'random', 'zeros', or 'ones'.")


### Defining Tensor computations ###
def func(a,b):
  
  c = tf.add(a, b)
  d = tf.subtract(b, 1)
  e = tf.multiply(c, d)

  return e



def main():

      # 0-d Tensors
      sport = tf.constant("Tennis", tf.string)
      number = tf.constant(1.41421356237, tf.float64)

      print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
      print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

      # 1-d Tensors
      sports = tf.constant(["Tennis", "Basketball"], tf.string)
      numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

      print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
      print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))
      

      ### Defining higher-order Tensors ###

      # 2-d Tensor
      tensor_shape = (3, 5)      
      matrix = get_nd_tensor(tensor_shape, "random")
      print(matrix)

      assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
      assert tf.rank(matrix).numpy() == 2

      row_vector = matrix[1]
      column_vector = matrix[:,1]
      scalar = matrix[0, 1]

      print("`row_vector`: {}".format(row_vector.numpy()))
      print("`column_vector`: {}".format(column_vector.numpy()))
      print("`scalar`: {}".format(scalar.numpy()))
      

      # Use tf.zeros to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
      #   You can think of this as 10 images where each image is RGB 256 x 256.
      # 4-d Tensor
      tensor_shape = (10, 256, 256, 3)      
      images = get_nd_tensor(tensor_shape, "zeros")
      #print(images)

      assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
      assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
      assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
      

      # Create the nodes in the graph, and initialize values
      a = tf.constant(15)
      b = tf.constant(61)

      # Add them!
      c1 = tf.add(a,b)
      c2 = a + b # TensorFlow overrides the "+" operation so that it is able to act on Tensors
      print(c1)
      print(c2)


      # Consider example values for a,b
      a, b = 1.5, 2.5
      # Execute the computation
      e_out = func(a,b)
      print(e_out)



if __name__ == "__main__":
    main()      
