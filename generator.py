import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import numpy as np
import dataset

import matplotlib.pyplot as plt
from skimage import io


#################################################################################
# Globals
#################################################################################

#number of samples per iteration
m = 200

#epochs updating D
k = 1

#epochs
epochs = 500

batch_size = 25

#################################################################################
# Discrimnator
#################################################################################

def discriminator_loss( y_true, y_pred ):
    one = y_true * K.log( K.clip( y_pred, K.epsilon(), None ) )
    two = ( 1. - y_true ) * K.log( 1. - K.clip ( y_pred, K.epsilon(), None )  )
    return K.mean( one  + two , axis = -1 )

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

    # model.compile( loss = discriminator_loss,
    #           optimizer = 'adam',
    #           metrics = [ 'accuracy' ] )

    optimizer = RMSprop(lr=0.0002, decay=6e-8)
    model.compile( loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'] )

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

    # model.compile( loss = generator_loss,
    #           optimizer = 'adam',
    #           metrics = [ 'accuracy' ] )

    optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)

    model.compile( loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'] )
    return model


#################################################################################
# Helpers
#################################################################################

def _generate_samples( model, num_samples ):
    noise = np.random.random_sample( ( num_samples, 100 ) )
    x_gen = model.predict( noise )
    y_gen = np.arange( x_gen.shape[ 0 ] )
    y_gen = np.zeros_like( y_gen )

    return x_gen, y_gen, noise

def stack_models( G, D ):
        model = Sequential()
        model.add( G )
        model.add( D )
        model.compile( loss = 'binary_crossentropy',
                        optimizer = RMSprop(lr=0.0001, decay=3e-8),
            metrics=['accuracy'])
        return model


#################################################################################
# Fun starts below
#################################################################################


(x_train, y_train), (x_test, y_test) = dataset.load()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

validation_noise = np.random.random_sample( ( 1, 100 ) )

D = build_discriminator( x_train.shape[ 1 : ] )
G = build_generator()

GD = stack_models( G, D )

#training epochs
for epoch in range( epochs ):
    print( 'Epoch: {} '.format( epoch ) )
    #train D
    for i in range( k ):
        #create m samples
        (x_gen, y_gen, noise) = _generate_samples( G, m )
        #merge with random samples from the true set
        np.random.shuffle( x_train ) #no need to shuffle y_train it is all ones anyway
        x = np.concatenate( (x_train[ : m ], x_gen) )
        y = np.concatenate( (y_train[ : m ], y_gen) )
        # loss = D.train_on_batch( x, y )
        # print( 'D loss: {}, acc: {}'.format( loss[ 0 ],  loss[ 1 ]  )  )
        D.fit( x = x, y = y, batch_size = 25, epochs = 1)

    #train G
    (x_gen, y_gen, noise) = _generate_samples( G, m )
    y_pred = D.predict( x_gen )
    for batch in np.array_split( noise, noise.shape[ 0 ] / batch_size  ):
        loss = GD.train_on_batch( batch, y_train[ : batch.shape[ 0 ] ] )

    print( 'D loss: {}, acc: {}'.format( loss[ 0 ],  loss[ 1 ]  )  )
    if epoch % 10 == 0:
        out = G.predict( validation_noise )
        out *= 255
        io.imsave( 'out/{}.png'.format( epoch ), out.astype(int)[ 0 ] )





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
