import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras import backend as K
import numpy as np
import dataset

import matplotlib.pyplot as plt


#################################################################################
# Globals
#################################################################################

#number of samples per iteration
m = 50

#epochs updating D
k = 1

#epochs
epochs = 200

#################################################################################
# Discrimnator
#################################################################################

def discriminator_loss( y_true, y_pred ):
    return K.mean( ( y_true * K.log( y_pred ) ) + ( 1 - y_true ) * K.log( 1 - y_pred  ) )

def build_discriminator( input_shape ):
    model = Sequential()
    model.add( Conv2D( 32, (3, 3), padding='same', input_shape = input_shape ) )
    model.add( Activation( 'relu' ) )
    model.add( Conv2D( 32, (3, 3) ) )
    model.add( Activation( 'relu') )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout( 0.25 ) )

    model.add( Conv2D( 64, (3, 3), padding='same' ) )
    model.add( Activation( 'relu' ) )
    model.add( Conv2D( 64, (3, 3) ) )
    model.add( Activation( 'relu' ) )
    model.add( MaxPooling2D( pool_size=( 2, 2 ) ) )
    model.add( Dropout(0.25) )

    model.add( Flatten() )
    model.add( Dense( 512 ) )
    model.add( Activation( 'relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense( 1 ) )

    model.compile( loss = discriminator_loss,
              optimizer = 'adam',
              metrics = [ 'accuracy' ] )
    return model


#################################################################################
# Generator
#################################################################################

gen_inputs = 100

def generator_loss( y_true, y_pred ):
    return K.mean( K.log( 1 - y_pred  ) )


# 53, 33, 3 sample shape

def build_generator():
    model = Sequential()
    model.add( Dense( gen_inputs , input_shape = (gen_inputs, ) ) )
    model.add( Activation( 'relu' ) )
    model.add( Dense( 300 ) )
    model.add( Activation( 'relu' ) )
    model.add( Dense( 312 ) )

    model.add( Reshape( ( 13, 8, 3  ) ) )
    model.add( Conv2DTranspose( 64, (3, 3), padding = 'same', strides= ( 2, 2 ) ) )
    model.add( Activation( 'relu' ) )
    model.add( Conv2DTranspose( 64, (3, 3), strides= ( 2, 2 ) ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2DTranspose( 64, (3, 3), padding = 'same' ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2DTranspose( 3, (3, 3), padding = 'same' ) )
    model.add( Activation( 'relu' ) )
    # model.add( Cropping2D( cropping=( ( 7 , 7 ), ( 1 , 2 ) ) ) )

    model.compile( loss = generator_loss,
              optimizer = 'adam',
              metrics = [ 'accuracy' ] )
    return model


(x_train, y_train), (x_test, y_test) = dataset.load()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


#random samples to feed to the Generator
z_train = np.random.random_sample( ( m, 100 ) )

D = build_discriminator( x.train.shape[ 1 : ] )
G = build_generator()






# if __name__ == '__main__':
#
#     gen = build_generator()
#     for layer in gen.layers:
#         print( layer.name )
#         print( layer.output_shape )
#
#     noises = np.random.random_sample( ( 5, 100 ) )
#     imgs = gen.predict( noises )
#
#     print( imgs.shape )
#
#     for i in imgs:
#         i *= 255
#         plt.imshow( i )
#         plt.show()
