#
# 2024 Compute Ontario Summer School
# Artificial Neural Networks
# Day 3, morning
# Erik Spence.
#
#
# This file, unet.py, contains the code which builds the neural
# network used as part of the diffusion network.
#

#######################################################################


import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.saving as ks


#######################################################################


@ks.register_keras_serializable(package = "MyLayers")
class Conv3Layer(kl.Layer):

    def __init__(self, num_out_channels, is_res = False, **kwargs):
        super().__init__(**kwargs)
        self.num_out_channels = num_out_channels
        self.Conv2D_1 = kl.Conv2D(self.num_out_channels,
                                  kernel_size = (3, 3),
                                  padding = 'same',
                                  activation = 'linear')
        self.GroupNorm_1 = kl.GroupNormalization(groups = 8)
        self.Conv2D_2 = kl.Conv2D(self.num_out_channels,
                                  kernel_size = (3, 3),
                                  padding = 'same',
                                  activation = 'linear')
        self.GroupNorm_2 = kl.GroupNormalization(groups = 8)
        self.is_res = is_res

        
    def call(self, inputs):
        x = self.Conv2D_1(inputs)
        x = self.GroupNorm_1(x)
        x1 = kl.ReLU()(x)

        x = self.Conv2D_2(x1)
        x = self.GroupNorm_2(x)
        x = kl.ReLU()(x)

        if self.is_res: x = (x1 + x)
        return x

    
#######################################################################


@ks.register_keras_serializable(package = "MyLayers")
class UnetDownLayer(kl.Layer):

    def __init__(self, num_out_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_out_channels = num_out_channels
        self.Conv3 = Conv3Layer(num_out_channels)
        self.MaxPool2D = kl.MaxPool2D(pool_size = (2, 2),
                                      strides = 2)

    def call(self, inputs):
        x = self.Conv3(inputs)
        x = self.MaxPool2D(x)
        return x

    
#######################################################################


@ks.register_keras_serializable(package="MyLayers")
class UnetUpLayer(kl.Layer):

    def __init__(self, num_out_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_out_channels = num_out_channels
        self.Conv2DTranspose = kl.Conv2DTranspose(num_out_channels,
                                                  kernel_size = (2, 2),
                                                  strides = 2,
                                                  activation = 'relu')
        self.Conv3_1 = Conv3Layer(num_out_channels)
        self.Conv3_2 = Conv3Layer(num_out_channels)

    def call(self, inputs):

        img_input = inputs[0]
        skip = inputs[1]
        
        x = tf.concat([img_input, skip], -1)
        x = self.Conv2DTranspose(x)
        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        return x

    
#######################################################################    


@ks.register_keras_serializable(package="MyLayers")
class TimeSirenLayer(kl.Layer):

    def __init__(self, emb_dim, **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.Dense_1 = kl.Dense(emb_dim, activation = 'linear')
        self.Dense_2 = kl.Dense(emb_dim, activation = 'linear')

    def call(self, inputs):

        x = self.Dense_1(inputs)
        x = tf.math.sin(x)
        x = self.Dense_2(x)
        return x

    
#######################################################################


def NaiveUnet(input_shape, n_feat):

    ## This image input.
    img_input = kl.Input(shape = input_shape)

    ## The time input
    t_input = kl.Input(shape = (1,))

    ## This is init_conv.
    x = Conv3Layer(n_feat, is_res = True)(img_input)
    
    down1 = UnetDownLayer(n_feat)(x)
    down2 = UnetDownLayer(2 * n_feat)(down1)
    down3 = UnetDownLayer(2 * n_feat)(down2)

    ## This is to_vec.
    thro = kl.AveragePooling2D(pool_size = (4, 4))(down3)
    thro = kl.ReLU()(thro)
    
    timeemb = TimeSirenLayer(2 * n_feat)(t_input)
    timeemb = kl.Reshape((1, 1, 2 * n_feat))(timeemb)

    ## This is up0.
    thro = kl.Conv2DTranspose(2 * n_feat, kernel_size = (4, 4),
                              strides = 4,
                              activation = 'linear')(thro + timeemb)
    thro = kl.GroupNormalization(groups = 8)(thro)
    thro = kl.ReLU()(thro)

    up1 = UnetUpLayer(2 * n_feat)([thro, down3])
    up2 = UnetUpLayer(n_feat)([up1, down2])
    up3 = UnetUpLayer(n_feat)([up2, down1])

    out = kl.Concatenate(axis = -1)([up3, x])
    out = kl.Conv2D(3, kernel_size = (3, 3),
                    padding = 'same',
                    activation = 'linear')(out)
    
    return km.Model(inputs = [img_input, t_input],
                    outputs = out)


#######################################################################    
