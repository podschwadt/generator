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

import warnings
import os


#################################################################################
# Globals
#################################################################################

RUN = 11

#number of samples per iteration
m = 200

#epochs updating D
k = 1

#epochs
epochs = 15000

#batch_size
batch_size = 25

# optimizers
d_opt = ( Adam(lr=0.000002, decay=6e-8), 'binary_crossentropy' , ['accuracy'] )
g_opt = ( Adam(lr=0.00008, clipvalue=1.0, decay=6e-8) , 'binary_crossentropy' , ['accuracy'] )
stacked_opt = ( Adam(lr=0.00008, clipvalue=1.0, decay=3e-8), 'binary_crossentropy' , ['accuracy'] )

g_acc_history = 10



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
    model.add( Activation( 'relu' ) )

    # model.compile( loss = discriminator_loss,
    #           optimizer = 'adam',
    #           metrics = [ 'accuracy' ] )


    model.compile( loss=d_opt[ 1 ],
                    optimizer=d_opt[ 0 ],
                    metrics=d_opt[ 2 ] )

    return model


#################################################################################
# Generatordirector
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
    # model.add( Dense( 378 ) )
    # model.add( Dropout( 0.4 ) )
    #
    # model.add( Reshape( ( 14, 9, 3  ) ) )
    model.add( Dense( 312 ) )
    model.add( Dropout( 0.4 ) )

    model.add( Reshape( ( 13, 8, 3  ) ) )
    model.add( Conv2DTranspose( 64
                                ,(3, 3)
                                ,strides= ( 2, 2 )
                                ,padding = 'same'
                                ) )
    model.add( Activation( 'relu' ) )
    model.add( Conv2DTranspose( 64
                                ,(4, 3)
                                , padding = 'same'
                                ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2DTranspose( 64
                                ,(3, 3)
                                ,strides= ( 2, 2 )
                                ,padding = 'same'
                                ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2DTranspose( 64
                                ,(2, 2)
                                ,strides=(1, 1)
                                ,padding = 'same'
                                ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2DTranspose( 3
                                ,(2, 2)
                                # ,padding = 'same'
                                ) )
    model.add( Activation( 'relu' ) )
    # model.add( Cropping2D( cropping=( ( 7 , 7 ), ( 1 , 2 ) ) ) )

    # model.compile( loss = generator_loss,
    #           optimizer = 'adam',
    #           metrics = [ 'accuracy' ] )


    model.compile( loss=g_opt[ 1 ],
                    optimizer=g_opt[ 0 ],
                    metrics=g_opt[ 2 ]  )
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
        D.trainable = False
        model.add( G )
        model.add( D )
        model.compile( loss=stacked_opt[ 1 ],
                        optimizer=stacked_opt[ 0 ],
                        metrics=stacked_opt[ 2 ] )
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


print('Discrimnator:\n')
D.summary()
print('Generator:\n')
G.summary()

GD = stack_models( G, D )


if not os.path.exists( 'out/' ):
    os.makedirs( 'out/' )


if not os.path.exists( 'report/run{}'.format( RUN ) ):
    os.makedirs( 'report/run{}'.format( RUN )  )

acc_list = range( g_acc_history )
index  = 0

train_D = True

#training epochs
msg = 'Epoch: {};  D loss: {}, acc: {}; G loss: {}, acc: {} '
for epoch in range( epochs ):
    # print( 'Epoch: {} '.format( epoch ) )
    #train D
    if train_D:
        for i in range( k ):
            #create m samples
            (x_gen, y_gen, noise) = _generate_samples( G, m )
            #merge with random samples from the true set
            np.random.shuffle( x_train ) #no need to shuffle y_train it is all ones anyway
            x = np.concatenate( (x_train[ : m ], x_gen) )
            y = np.concatenate( (y_train[ : m ], y_gen) )
            # loss = D.train_on_batch( x, y )
            # print( 'D loss: {}, acc: {}'.format( loss[ 0 ],  loss[ 1 ]  )  )
            history = D.fit( x = x, y = y, batch_size = 25, epochs = 1, verbose = 0 ).history
            # print( 'D loss: {}, acc: {}'.format( history.history[ 'acc' ][ 0 ], history.history[ 'loss' ][ 0 ]  )  )


    #train G
    # D.trainable = False
    (x_gen, y_gen, noise) = _generate_samples( G, m )
    y_pred = D.predict( x_gen )
    l = 0.0
    a = 0.0
    steps = noise.shape[ 0 ] / batch_size
    for batch in np.array_split( noise, noise.shape[ 0 ] / batch_size  ):
        loss = GD.train_on_batch( batch, y_train[ : batch.shape[ 0 ] ] )
        l += loss[ 0 ]
        a += loss[ 1 ]
    # D.trainable = True

    #look at the last X losses and decide if we train D
    acc_list[ index ] = a / steps
    index %=  g_acc_history
    if( ( sum( acc_list ) / g_acc_history ) < 0.01 ):
        train_D = False
    else:
        train_D = True

    #report every 10 epochs
    if epoch % 10 == 0:
        print( msg.format(epoch, history[ 'acc' ][ 0 ], history[ 'loss' ][ 0 ]  , l / steps, a / steps  ) )
        out = G.predict( validation_noise ) * 255
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave( 'out/{}.png'.format( epoch ), out[0].astype('uint8'), plugin='pil' )
            io.imsave( 'report/run{}/{}.png'.format( RUN, epoch ), out[0].astype('uint8'), plugin='pil' )





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
