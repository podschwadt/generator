from __future__ import division
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, AveragePooling2D, Dense, BatchNormalization
from keras.layers import Activation, AveragePooling2D
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.layers import Conv2D
import keras.backend as K
import numpy as np


def normalize( data, max_value=None, minus_one=True  ):
    if max_value is None:
        mx = np.max( data )
    else:
        mx = max_value
    if minus_one:
        return ( data / ( mx / 2.0 ) ) -1
    else:
        return data / ( 1.0 * mx )

def cifar10_0():

    (data, _ ), ( _ , _ ) = cifar10.load_data()
    #discriminator

    data = normalize( data, 255 )

    nch = 256
    D = Sequential()
    h = 5
    print data.shape
    D.add(Conv2D(int(nch / 4), h, strides=h, padding='same', input_shape=data.shape[ 1 : ] ) )
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add(Conv2D(int(nch / 2), h, strides=h, padding='same'))
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add( Conv2D(nch, h, strides=h, padding='same'))
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add(Conv2D(1, h, strides=h, padding='same'))
    # D.add(AveragePooling2D(pool_size=(4, 4), padding='valid'))
    D.add(Flatten())
    D.add(Activation('sigmoid'))
    D.summary()

    #generator

    G = Sequential()
    nch = 256
    h = 5
    G.add( Dense( nch * 4 * 4, input_dim=100 ) )
    # G.add( BatchNormalization() )
    G.add( Reshape( ( ( 4, 4, nch ) )) )
    G.add( Conv2D( int(nch / 2), (5,5), padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 2 ), h, padding='same') )
    # G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 4), h, padding='same') )
    # G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=(2, 2) ) )
    G.add( Conv2D( 3, h, padding='same') )
    G.add( Activation( 'sigmoid' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data


def cifar10_1():

    (data, _ ), ( _ , _ ) = cifar10.load_data()
    #discriminator

    data = normalize( data, 255 )

    nch = 512
    D = Sequential()
    h = 5

    D.add( Conv2D( nch, h, strides=2, padding='same', input_shape=data.shape[ 1 : ] ) )
    D.add( LeakyReLU( 0.2 ) )
    D.add( Conv2D( int( nch / 2 ), h, strides=2, padding='same'))
    D.add( LeakyReLU( 0.2 ) )
    D.add( Conv2D( nch, h, strides=2, padding='same' ) )
    D.add( LeakyReLU( 0.2 ) )
    D.add( Conv2D( int( nch / 2 ) , h, strides=2, padding='same' ) )
    D.add( LeakyReLU( 0.2 ) )
    D.add( Conv2D( 1, h, strides=2, padding='same' ) )
    D.add( LeakyReLU( 0.2 ) )
    D.add( Flatten() )
    D.add( Activation( 'sigmoid' ) )
    D.summary()

    #generator

    G = Sequential()
    nch = 512
    h = 5
    G.add( Dense( nch * 4 * 4, input_dim=100 ) )
    G.add( Reshape( ( ( 4, 4, nch ) ) ) )
    G.add( Conv2D( int(nch / 2), (5,5), padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 2 ), h, padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 4), h, padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=(2, 2) ) )
    G.add( Conv2D( 3, h, padding='same') )
    G.add( Activation( 'sigmoid' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data


def cifar10_2( input_dim=100 ):

    (data, _ ), ( _ , _ ) = cifar10.load_data()
    #discriminator

    data = normalize( data, 255 )

    nch = 512
    D = Sequential()
    h = 5

    channels_first = K.image_data_format() == 'channels_first'

    D.add( Conv2D( nch, h, strides=2, padding='same', input_shape=data.shape[ 1 : ] ) )
    D.add( Activation( 'relu' ) )
    D.add( Conv2D( int( nch / 2 ), h, strides=2, padding='same'))
    D.add( Activation( 'relu' ) )
    D.add( Conv2D( nch, h, strides=2, padding='same' ) )
    D.add( Activation( 'relu' ) )
    D.add( Conv2D( int( nch / 2 ) , h, strides=2, padding='same' ) )
    D.add( Activation( 'relu' ) )
    D.add( Conv2D( 1, h, strides=2, padding='same' ) )
    D.add( Activation( 'relu' ) )
    D.add( Flatten() )
    D.add( Activation( 'tanh' ) )
    D.summary()

    #generator

    G = Sequential()
    nch = 512
    h = 5
    G.add( Dense( nch * 4 * 4, input_dim=input_dim ) )
    G.add( Reshape(  ( nch, 4, 4 ) if channels_first else ( 4, 4, nch ) ) )
    G.add( Conv2D( int(nch / 2), (5,5), padding='same') )
    G.add( Activation( 'relu' ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 2 ), h, padding='same') )
    G.add( Activation( 'relu' ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 4), h, padding='same') )
    G.add( Activation( 'relu' ) )
    G.add( UpSampling2D( size=(2, 2) ) )
    G.add( Conv2D( 3, h, padding='same') )
    G.add( Activation( 'tanh' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data

def cifar10_3( input_dim=100 ):

    (data, _ ), ( _ , _ ) = cifar10.load_data()
    #discriminator

    data = normalize( data, 255 )

    nch = 512
    D = Sequential()
    h = 5

    channels_first = K.image_data_format() == 'channels_first'

    D.add( Flatten( input_shape=data.shape[ 1 : ] ) )
    D.add( Dense( 3072 ) )
    D.add( Activation( 'relu' ) )
    D.add( Dense( 1500 ) )
    D.add( Activation( 'relu' ) )
    # D.add( Dense( 750 ) )
    # D.add( Activation( 'relu' ) )
    # D.add( Dense( 512 ) )
    # D.add( Activation( 'relu' ) )
    D.add( Dense( 256 ) )
    D.add( Activation( 'relu' ) )
    D.add( Dense( 128 ) )
    D.add( Activation( 'relu' ) )
    D.add( Dense( 1 ) )
    D.add( Activation( 'tanh' ) )
    D.summary()

    #generator

    G = Sequential()
    nch = 512
    h = 5
    G.add( Dense( 200, input_dim=input_dim ) )
    G.add( Activation( 'relu' ) )
    G.add( Dense( 400 ) )
    G.add( Activation( 'relu' ) )
    # G.add( Dense( 800 ) )
    # G.add( Activation( 'relu' ) )
    # G.add( Dense( 1600 ) )
    # G.add( Activation( 'relu' ) )
    G.add( Dense( 3072 ) )
    G.add( Activation( 'relu' ) )
    print( D.layers[ -1 ].output_shape )
    G.add( Reshape( ( 3, 32, 32 ) if channels_first else ( 32, 32, 3 ) ) )
    G.add( Activation( 'tanh' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data


def cifar10_iwgan_paper( input_dim=128, filters=128 ):

    (data, _ ), ( _ , _ ) = cifar10.load_data()

    data = normalize( data, 255 )
    channels_first = K.image_data_format() == 'channels_first'

    assert input_dim % filters == 0

    resize = input_dim // filters

    activation = LeakyReLU( 0.2 )
    # Activation( 'relu' )


    #discriminator
    D = Sequential()

    D.add( Conv2D( filters, ( 3, 3 ), strides=2, input_shape=data.shape[ 1 : ], padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), strides=2, padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), padding='same' ) )
    D.add( activation )

    D.add( AveragePooling2D( pool_size=(8, 8), padding='same' ) )
    D.add( Flatten() )
    D.add( Dense( 1 ) )
    D.add( Activation( 'tanh' ) )

    D.summary()

    #generator
    G = Sequential()

    G.add( Dense( input_dim * 4 * 4, input_dim=input_dim ) )
    G.add( Reshape( ( resize * filters, 4, 4 ) if channels_first else ( 4, 4, resize * filters ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( 3, (3,3), padding='same') )
    G.add( Activation( 'tanh' ) )

    G.summary()


    return D, G, data

def cifar10_iwgan_paper_1( input_dim=128, filters=128, **kwargs ):
    """
    modified version adapted from  https://github.com/farizrahman4u/keras-contrib/blob/master/examples/improved_wgan.py
    """

    (data, _ ), ( _ , _ ) = cifar10.load_data()

    data = normalize( data, 255 )
    channels_first = K.image_data_format() == 'channels_first'

    assert input_dim % filters == 0

    resize = input_dim // filters

    if 'activation' in kwargs.keys():
        activation = kwargs[ 'activation' ]
    else:
        activation = LeakyReLU( 0.2 )

    normalize_axis = 1 if channels_first else  -1

    #discriminator
    D = Sequential()

    D.add( Conv2D( filters, ( 3, 3 ), strides=2, input_shape=data.shape[ 1 : ], padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), strides=2, padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), padding='same' ) )
    D.add( activation )

    D.add( Conv2D( filters, ( 3, 3 ), padding='same' ) )
    D.add( activation )

    D.add( AveragePooling2D( pool_size=(8, 8), padding='same' ) )
    D.add( Flatten() )
    D.add( Dense( 1 ) )
    D.add( Activation( 'tanh' ) )

    D.summary()

    #generator
    G = Sequential()

    G.add( Dense( input_dim * 4, input_dim=input_dim ) )
    G.add( activation )

    G.add( Dense( input_dim * 8 ) )
    G.add( BatchNormalization() )
    G.add( activation )

    G.add( Reshape( ( resize * filters, 4, 4 ) if channels_first else ( 4, 4, resize * filters ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( BatchNormalization( axis=normalize_axis ) )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( BatchNormalization( axis=normalize_axis ) )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( filters, (3,3), padding='same') )
    G.add( BatchNormalization( axis=normalize_axis ) )
    G.add( activation )
    G.add( UpSampling2D( size=( 2, 2 ) ) )

    G.add( Conv2D( 3, (3,3), padding='same') )
    G.add( Activation( 'tanh' ) )

    G.summary()


    return D, G, data

def mnist_0():

    (data, _ ), ( _ , _ ) = mnist.load_data()
    data = np.expand_dims( data, 1)
    data = normalize( data )

    nch = 256
    D = Sequential()
    h = 5
    print data.shape
    D.add(Conv2D(int(nch / 4), h, strides=h, padding='same', input_shape=data.shape[ 1 : ] ) )
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add(Conv2D(int(nch / 2), h, strides=h, padding='same'))
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add( Conv2D(nch, h, strides=h, padding='same'))
    # D.add(MaxPooling2D(pool_size=(2, 2)))
    D.add(LeakyReLU(0.2))
    D.add(Conv2D(1, h, strides=h, padding='same'))
    # D.add(AveragePooling2D(pool_size=(4, 4), padding='valid'))
    D.add(Flatten())
    D.add(Activation('sigmoid'))
    D.summary()

    #generator

    G = Sequential()
    nch = 256
    h = 5
    G.add( Dense( nch * 4 * 4, input_dim=100 ) )
    # G.add( BatchNormalization() )
    G.add( Reshape( ( ( 4, 4, nch ) )) )
    G.add( Conv2D( int(nch / 2), (5,5), padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 2 ), h, padding='same') )
    # G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 4), h, padding='same') )
    # G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=(2, 2) ) )
    G.add( Conv2D( 3, h, padding='same') )
    G.add( Activation( 'sigmoid' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data
