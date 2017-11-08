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

def loss( y_true, y_pred ):
    return K.mean( y_true * y_pred )

RUN = 11

#number of samples per iteration
m = 200

#epochs updating D
k = 20

#epochs
epochs = 1000

#batch_size
batch_size = 25

# optimizers
optimizer_d = ( Adam( lr=0.00005 ), loss  )
optimizer_stacked = ( Adam( lr=0.00005 ), loss )


#global helpers
no_out = True


#################################################################################
# Discrimnator
#################################################################################


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

    model.compile( loss=optimizer_d[ 1 ],
                    optimizer=optimizer_d[ 0 ]
                 )

    return model


#################################################################################
# Generatordirector
#################################################################################

gen_inputs = 100

# 53, 33, 3 sample shape

def build_generator():
    model = Sequential()
    model.add( Dense( gen_inputs , input_shape = (gen_inputs, ) ) )
    model.add( Activation( 'relu' ) )
    model.add( Dense( 300 ) )
    model.add( Activation( 'relu' ) )
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

    return model


#################################################################################
# Helpers
#################################################################################

def _generate_samples( model, num_samples ):
    noise = np.random.random_sample( ( num_samples, 100 ) )
    x_gen = model.predict( noise )
    y_gen = np.arange( x_gen.shape[ 0 ] )
    y_gen = np.ones_like( y_gen )
    return x_gen, y_gen, noise

def stack_models( G, D ):
        model = Sequential()
        D.trainable = False
        model.add( G )
        model.add( D )
        model.compile( loss=optimizer_stacked[ 1 ],
                        optimizer=optimizer_stacked[ 0 ]
                        )
        return model



#################################################################################
# Fun starts below
#################################################################################


(x_train, y_train), (x_test, y_test) = dataset.load()
x_train =  x_train.astype('float32') / 255.
x_test =  x_test.astype('float32')  / 255.
y_train *= -1.
y_test *= -1.

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

if not no_out:
    if not os.path.exists( 'report/run{}'.format( RUN ) ):
        os.makedirs( 'report/run{}'.format( RUN )  )


#training epochs
for epoch in range( epochs ):
    print( 'Epoch: {} '.format( epoch ) )
    #train D

    loss_D_real = []
    loss_D_fake = []

    if epoch == 100:
        k = k/2

    for i in range( k ):
        # unfreeze D
        D.trainable = True
        for l in D.layers:
            l.trainable = True

        #create m samples
        (x_gen, y_gen, noise) = _generate_samples( G, m )
        #merge with random samples from the true set
        np.random.shuffle( x_train ) #no need to shuffle y_train it is all ones anyway
        x = np.concatenate( (x_train[ : m ], x_gen) )
        y = np.concatenate( (y_train[ : m ], y_gen) )
        # loss = D.train_on_batch( x, y )
        loss_D_real.append( D.fit( x = x_train[ : m ], y = y_train[ : m ], batch_size = 25, epochs = 1, verbose = 0 ).history[ 'loss' ][ 0 ] )
        loss_D_fake.append( D.fit( x = x_gen, y = y_gen, batch_size = 25, epochs = 1, verbose = 0 ).history[ 'loss' ][ 0 ] )

        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            l.set_weights(weights)

        if i % 20 == 0:
            print( 'D {}/{} loss real/fake: {}/{}'.format( i + 1, k, sum( loss_D_real ) / len( loss_D_real ), sum( loss_D_fake ) / len( loss_D_fake ) ) )


    #train G
    D.trainable = False
    for l in D.layers:
        l.trainable = False
    (x_gen, y_gen, noise) = _generate_samples( G, m )
    l = 0.0
    steps = noise.shape[ 0 ] / batch_size
    for batch in np.array_split( noise, noise.shape[ 0 ] / batch_size  ):
        loss = GD.train_on_batch( batch, y_train[ : batch.shape[ 0 ] ] )
        l += loss



    print( 'G loss: {} '.format(  l / steps  ) )
    out =  G.predict( validation_noise ) *255
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave( 'out/{}.png'.format( epoch ), out[0].astype('uint8'), plugin='pil' )
        if not no_out:
            io.imsave( 'report/run{}/{}.png'.format( RUN, epoch ), out[0].astype('uint8'), plugin='pil' )

D.save( 'saved_models/wgan_D_{}.h5'.format( RUN ) )
G.save( 'saved_models/wgan_G_{}.h5'.format( RUN ) )



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
