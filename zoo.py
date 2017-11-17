from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, AveragePooling2D, Dense, BatchNormalization
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.layers import Conv2D


def cifar10_0():

    (data, _ ), ( _ , _ ) = cifar10.load_data()
    #discriminator

    nch = 256
    D = Sequential()
    h = 5
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
    G.add( BatchNormalization() )
    G.add( Reshape( ( ( 4, 4, nch ) )) )
    G.add( Conv2D( int(nch / 2), (5,5), padding='same') )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 2 ), h, padding='same') )
    G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=( 2, 2 ) ) )
    G.add( Conv2D( int( nch / 4), h, padding='same') )
    G.add( BatchNormalization( axis=1 ) )
    G.add( LeakyReLU( 0.2 ) )
    G.add( UpSampling2D( size=(2, 2) ) )
    G.add( Conv2D( 3, h, padding='same') )
    G.add( Activation( 'sigmoid' ) )

    G.summary()
    print(G.get_input_shape_at( 0 ))

    return D, G, data
