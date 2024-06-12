#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 1

import plotting_routines as pr
import numpy as np

import example1 as ex1
import example2 as ex2
import mnist_loader

import first_network as fn1
import second_network as fn2


def main() -> None:

    example = 3

    if example == 1:
        # Generate a sythetic dataset and
        # split it into training and testing datasets
        train_pos, test_pos, train_value, test_value = ex1.get_data(500)    

        print(
            train_pos.shape, train_value.shape, train_pos[0], train_value[0]
        )
        

        # Plot the training data
        pr.plot_dots(train_pos, train_value)

        # Plot the testing data
        pr.plot_dots(test_pos, test_value)

        # Build the NN model with a single neuron and
        # the sigmoid activation function
        model = fn1.build_model(train_pos, train_value, eta=5e-5)

        # Plot the decision boundary for the training data 
        # using the built NN model
        pr.plot_decision_boundary(train_pos, train_value, model, fn1.predict)

        # Get the prediction value for the testing data 
        # using the built NN model
        f = fn1.predict(test_pos, model)

        # The percentage of corrected prediction for the testing data
        print(sum(np.round(f) == test_value) / len(test_value)) * 100

    elif example == 2:

        train_pos, test_pos, train_value, test_value = ex2.get_data(500)

        print(
            train_pos.shape, train_value.shape, train_pos[0], train_value[0]
        )

        pr.plot_dots(train_pos, train_value)
        pr.plot_dots(test_pos, test_value)

        model = fn2.build_model(
            5, train_pos, train_value, eta=5e-3, output_dim=2
        )
        # I increased the number of neurons from 10 to 20, 
        # and the training precsion increased from 82% to 85.2%,
        # but the test precision droped from 86% to 72%.
        # This is the overfitting problem.
        # I set the number of neurons as 5, 
        # and precision of training and testing are 83.5% and 87%

        pr.plot_decision_boundary(train_pos, train_value, model, fn2.predict)

        f = fn2.predict(test_pos, model)

        print(sum(f == test_value) / len(test_value) * 100)

    else:

        train_x, test_x, train_y, test_y = mnist_loader.\
                                    load_mnist_1D_small("mnist.pkl.gz")

        print(train_x.shape)

        n = 30
        model = fn2.build_model(
            n, train_x, train_y, output_dim=10, eta=5e-4, num_steps=20000
        )

        print(test_x.shape)

        f = fn2.predict(test_x, model)

        print(sum(f == test_y) / len(test_y) * 100)

        # The number of parameters in the model
        print((784+1)*n + (n+1)*10)


    return


if __name__ == "__main__":
    main()    