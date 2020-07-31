from tensorflow import  keras
import  tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2DTranspose, Lambda


# def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='valid'):
#     """
#         input_tensor: tensor, with the shape (batch_size, time_steps, dims)
#         filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
#         kernel_size: int, size of the convolution kernel
#         strides: int, convolution step size
#         padding: 'valid' | 'valid'
#     """
#     x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
#     x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
#     x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
#     return x

class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid',output_padding=None):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
          filters, (kernel_size, 1), (strides, 1), padding
        )

    def call(self, x):
        x = tf.expand_dims(x, axis=2)
        x = self.conv2dtranspose(x)
        x = tf.squeeze(x, axis=2)
        return x

class ECGDAE:
    def __init__(self,channel,siglength):
        self.channel=channel
        self.siglenth=siglength
        self.inputshape=(siglength,channel)
    def build(self):
        model=keras.Sequential()
        model.add(keras.layers.Conv1D(32,4,2,input_shape=self.inputshape))#filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(keras.layers.Conv1D(64, 4, 2))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(keras.layers.Conv1D(128, 4, 2))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(keras.layers.Conv1D(256, 4, 2))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(keras.layers.Conv1D(512, 4, 2))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(Conv1DTranspose(256, 4, 2,padding='valid' ))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(Conv1DTranspose(128, 5, 2,padding='valid'))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())

        model.add(Conv1DTranspose(64, 4, 2,padding='valid'))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(Conv1DTranspose(32, 4, 2,padding='valid'))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        model.add(Conv1DTranspose(1, 6, 2,padding='valid'))  # filters,kernel_size,strides
        model.add(keras.layers.PReLU())
        return  model



if __name__ == '__main__':
    # InputData=np.random.rand(1,1000)
    # print(InputData.shape)
    print(tf.__version__)
    ECGDAEModel = ECGDAE(1, 1000).build()
    ECGDAEModel.summary()
