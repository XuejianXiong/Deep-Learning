#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 3

from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as ku
import tensorflow.keras.backend as K

import model1 as m1



def main() -> None:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)
    print(x_test.shape)

    model_i = 5

    if model_i == 1 or model_i == 2:

        x_train2 = x_train[0:500, :, :].reshape(500, 784)
        x_test2 = x_test[0:100, :, :].reshape(100, 784)
        y_train2 = ku.to_categorical(y_train[0:500], 10)
        y_test2 = ku.to_categorical(y_test[0:100], 10)

    elif model_i == 3:

        x_train2 = x_train.reshape(60000, 784)
        x_test2 = x_test.reshape(10000, 784)
        y_train2 = ku.to_categorical(y_train, 10)
        y_test2 = ku.to_categorical(y_test, 10)

    elif model_i == 4 or model_i == 5:

        print(K.image_data_format())

        x_train2 = x_train.reshape(60000, 28, 28, 1)
        x_test2 = x_test.reshape(10000, 28, 28, 1)
        y_train2 = ku.to_categorical(y_train, 10)
        y_test2 = ku.to_categorical(y_test, 10)

    print(y_test2.shape)
    #print(y_test2[0:10,:])
    #print(y_test.shape)
    #print(y_test[0:10])

    if model_i == 1: 
        model = m1.get_model(30, 784, 10)
    elif model_i == 2:
        model = m2.get_model(30, d_rate=0.2)
    elif model_i == 3:
        model = m3.get_model(30)
    elif model_i == 4 or model_i == 5:
        model = m4.get_model(20, 100)
    

    print(model.output_shape)
    print(model.summary())

    if model_i == 1 or model_i == 2:
        model.compile(
            optimizer="sgd", 
            metrics=["accuracy"], 
            loss="mean_squared_error"
        )
        
        fit = model.fit(
            x_train2, 
            y_train2, 
            epochs=1000, 
            batch_size=5, 
            verbose=0
        )

    elif model_i == 3:

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        fit = model.fit(
            x_train2, 
            y_train2, 
            epochs=100, 
            batch_size=128, 
            verbose=2
        )
    elif model_i == 4:

        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"]
        )

        fit = model.fit(
            x_train2, 
            y_train2, 
            epochs=30, 
            batch_size=128, 
            verbose=2
        )
    
    elif model_i == 5:

        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"]
        )

        fit = model.fit(
            x_train2, 
            y_train2, 
            epochs=100, 
            batch_size=128, 
            verbose=2
        )




    # score = [the value of the loss function, accuracy]
    score = model.evaluate(x_test2, y_test2)
    print(score)

    return


if __name__ == "__main__":
    main()    