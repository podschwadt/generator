import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import keras.datasets
import numpy as np
import dataset

import matplotlib.pyplot as plt
from skimage import io

import warnings
import os

from tools import CircleList, image_grid
import time





#################################################################################
# Globals
#################################################################################

def loss( y_true, y_pred ):
    return K.mean( y_true * y_pred )

RUN = 2

#number of samples per iteration
m = 200

#epochs updating D
k = 100

#epochs
epochs = 500

#batch_size
batch_size = 25

# optimizers
lr = 0.00005
optimizer_d = ( RMSprop( lr=lr ), loss  )
optimizer_stacked = ( RMSprop( lr=lr ), loss )

#threshold after we stop learning
threshold = 0.001

#generator stuff
gen_inputs = 100

#global helpers
no_out = True

def print_summary():
    msg = """
    #################################################################################
    # {}
    #################################################################################

    def loss( y_true, y_pred ):
        return K.mean( y_true * y_pred )

    #number of samples per iteration
    m = {}

    #epochs updating D
    k = {}

    #epochs
    epochs = {}

    #batch_size
    batch_size = {}

    # optimizers
    optimizer_d = ({}( lr={:.2e} ), loss  )
    optimizer_stacked = ( {}( lr={:.2e} ), loss )

    #threshold after we stop learning
    threshold = {}

    """
    print( msg.format( RUN, m, k, epochs, batch_size, optimizer_d.__class__.__name__, lr, optimizer_stacked.__class__.__name__, lr, threshold ) )

#################################################################################
# Discrimnator
#################################################################################


def build_discriminator( input_shape ):
    model = Sequential()
    model.add( Conv2D( 32, (3, 3), padding='same', input_shape = ( 28,28,1 ) ) )
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
# Generator
#################################################################################

gen_inputs = 100

# 53, 33, 3 sample shape

def build_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation='relu', input_shape=(gen_inputs,) ) )
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))
    # model = Sequential()
    # model.add( Dense( gen_inputs , input_shape = (gen_inputs, ) ) )
    # model.add( Activation( 'relu' ) )
    # model.add( Dense( 300 ) )
    #
    #
    # model.add( Activation( 'relu' ) )
    # model.add( Dense( 256 ) )
    # model.add( Activation( 'relu' ) )
    # model.add( Dense( 49 ) )
    # model.add( Dropout( 0.4 ) )
    #
    # model.add( Reshape( ( 7, 7,1 ) ) )
    # model.add( Conv2DTranspose( 64
    #                             ,(3, 3)
    #                             ,strides= ( 2, 2 )
    #                             ,padding = 'same'
    #                             ) )
    # model.add( Activation( 'relu' ) )
    # model.add( Conv2DTranspose( 64
    #                             ,(4, 3)
    #                             , padding = 'same'
    #                             ) )
    # model.add( Activation( 'relu' ) )
    #
    # model.add( Conv2DTranspose( 64
    #                             ,(2, 2)
    #                             ,strides= ( 2, 2 )
    #                             , padding = 'valid'
    #                             ) )
    # model.add( Activation( 'relu' ) )
    #
    # model.add( Conv2DTranspose( 1
    #                             ,(2, 2)
    #                             ,strides=(1, 1)
    #                             ,padding = 'same'
    #                             ) )
    # model.add( Activation( 'relu' ) )

    return model


#################################################################################
# Helpers
#################################################################################

def _generate_samples( model, num_samples ):
    noise = np.random.random_sample( ( num_samples, gen_inputs ) )
    x_gen = model.predict( noise )
    y_gen = np.arange( x_gen.shape[ 0 ] )
    y_gen = np.ones_like( y_gen )

    x_gen = x_gen.reshape(x_gen.shape[0], 28, 28,1)


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


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train =  x_train.astype('float32') / 255.
x_test =  x_test.astype('float32')  / 255.
y_train = np.ones_like(  np.arange( x_train.shape[ 0 ] ) ).astype( 'float32' ) * -1.

x_train = x_train.reshape(x_train.shape[0],  28, 28 ,1)



validation_noise = np.random.random_sample( ( 25, gen_inputs ) )

D = build_discriminator( x_train.shape[ 1 : ] )
G = build_generator()


print_summary()
print('nDiscrimnator:\n')
D.summary()
print('\nGenerator:\n')
G.summary()

GD = stack_models( G, D )


if not os.path.exists( 'out/' ):
    os.makedirs( 'out/' )

if not no_out:
    if not os.path.exists( 'report/run_wgan_{}'.format( RUN ) ):
        os.makedirs( 'report/run_wgan_{}'.format( RUN )  )


#training epochs
for epoch in range( epochs ):
    print( 'Epoch: {} '.format( epoch ) )
    start_t = time.time()
    #train D

    loss_D_real = []
    loss_D_fake = []

    if epoch == 100:
        k = k/2

    l_real = CircleList( 5 )
    l_fake = CircleList( 5 )

    num_k = 0

    for i in range( k ):

        # unfreeze D
        D.trainable = True
        for l in D.layers:
            l.trainable = True

        #create m samples
        (x_gen, y_gen, noise) = _generate_samples( G, m )
        np.random.shuffle( x_train ) #no need to shuffle y_train it is all ones anyway


        loss = D.fit( x = x_train[ : m ], y = y_train[ : m ], batch_size = 25, epochs = 1, verbose = 0 ).history[ 'loss' ][ 0 ]
        loss_D_real.append( loss )
        l_real.add( loss )

        loss = D.fit( x = x_gen, y = y_gen, batch_size = 25, epochs = 1, verbose = 0 ).history[ 'loss' ][ 0 ]
        loss_D_fake.append( loss )
        l_fake.add( loss )



        #clip weights becuase MATH
        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            l.set_weights(weights)

        # train D at least 5 epoch and stop improvement is small
        # make sure we train the full lenght for the first 5 outer loops
        # if i > 5 and epoch > 5:
        #    if l_real.variance() < threshold and l_fake.variance() < threshold:
        #        break


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



    print( 'D loss real/fake: {:.2e}/{:.2e} | trained for: {}'.format( sum( loss_D_real ) / len( loss_D_real ), sum( loss_D_fake ) / len( loss_D_fake ) , i ) )
    print( 'G loss: {} '.format(  l / steps  ) )
    print( 'Time: {} s'.format( time.time() - start_t ) )
    out =  G.predict( validation_noise ) * 255
    print( out.shape )
    out = image_grid( out, debug=True )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave( 'out/{}.png'.format( epoch ), out, plugin='pil' )
        if not no_out:
            io.imsave( 'report/run{}/{}.png'.format( RUN, epoch ), out, plugin='pil' )

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
