import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.layers.merge import _Merge
from keras.engine.topology import Layer
import keras.datasets
import numpy as np
import dataset

import matplotlib.pyplot as plt
from skimage import io

import warnings
import os

from tools import CircleList, image_grid
import time



# based on
# http://shaofanlai.com/post/10
# https://github.com/PiscesDream/Keras-GAN


#################################################################################
# Globals
#################################################################################

def multiply_mean( y_true, y_pred ):
    return K.mean( y_true * y_pred )

def mean( y_true, y_pred ):
    return K.mean( y_pred )

RUN = 2

#number of samples per iteration
m = 50

#epochs updating D
k = 10

#epochs
epochs = 500

#batch_size
batch_size = 25

# optimizers
optimizer = Adam( lr=0.001, beta_1=0.5, beta_2=0.9 ) # proposed values in the paper


optimizer_d = ( optimizer, 'binary_crossentropy'  )
optimizer_stacked = ( optimizer, multiply_mean )

#threshold after we stop learning
threshold = 0.001

#clipvalue
clipvalue = 0.01

true_label = -1.
fake_label = 1.

# lambda in the paper
penalty_coef = 10

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
# Helper classes
#################################################################################
class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output-inputs[i]
        return output


class GradNorm(Layer):
    def __init__( self, **kwargs ):
        super( GradNorm, self ).__init__( **kwargs )

    def build( self, input_shapes ):
        super( GradNorm, self ).build( input_shapes )

    def call( self, inputs ):
        target, wrt = inputs
        grads = K.gradients( target, wrt )
        assert len( grads ) == 1
        grad = grads[ 0 ]
        return K.sqrt( K.sum( K.batch_flatten( K.square( grad ) ), axis=1, keepdims=True ) )

    def compute_output_shape( self, input_shapes ):
        return ( input_shapes[ 1 ][ 0 ], 1 )



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

# 28, 28, 1 sample shape

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
    noise = np.random.uniform( -1., 1., size = ( num_samples, gen_inputs) ).astype( 'float32' )
    x_gen = model.predict( noise )
    y_gen = np.arange( x_gen.shape[ 0 ] )
    y_gen = np.ones_like( y_gen ) * fake_label

    x_gen = x_gen.reshape( x_gen.shape[ 0 ], 28, 28, 1 )


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
y_train = np.ones_like(  np.arange( x_train.shape[ 0 ] ) ).astype( 'float32' ) * true_label

x_train = x_train.reshape(x_train.shape[0],  28, 28 ,1)



validation_noise = np.random.uniform( -1., 1., size = ( 25, gen_inputs) ).astype( 'float32' )


D = build_discriminator( x_train.shape[ 1 : ] )
G = build_generator()


#print_summary()
print('nDiscrimnator:\n')
D.summary()
print('\nGenerator:\n')
G.summary()

GD = stack_models( G, D )


#more and funkier model combining and compiling
shape = D.get_input_shape_at(0)[1:]
gen_input, real_input, interpolation = Input( shape ), Input( shape ), Input( shape )
sub = Subtract()( [ D( gen_input ), D( real_input ) ] )
norm = GradNorm()( [ D(interpolation), interpolation ] )
mini_batch_model = Model ( [ gen_input, real_input, interpolation ], [ sub, norm ] )

mini_batch_model.compile( optimizer=optimizer, loss=[ mean, 'mse' ], loss_weights=[ 1.0, penalty_coef ] )


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

    d_loss, d_diff, d_norm  = [], [], []

    for i in range( k ):
        # unfreeze D
        D.trainable = True
        for l in D.layers:
            l.trainable = True

        #create m samples
        (x_gen, y_gen, noise) = _generate_samples( G, m )
        np.random.shuffle( x_train ) #no need to shuffle y_train it is all ones anyway

        epsilon = np.random.uniform( 0, 1, size=( m, 1, 1, 1 ) )
        interpolation = epsilon * x_train[ :m ] + ( 1 - epsilon ) * x_gen
        result = mini_batch_model.train_on_batch( [ x_gen, x_train[ :m ], interpolation] , [ np.ones( ( m, 1 ) ) ] * 2 )
        d_loss.append( result[ 0 ] )
        d_diff.append( result[ 1 ] )
        d_norm.append( result[ 2 ] )

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



    print( 'D loss/diff/norm: {:.2e}/{:.2e}/{:.2e} '.format( sum( d_loss ) / len( d_loss ), sum( d_diff ) / len( d_diff ) , sum( d_norm ) / len( d_norm ) ) )
    print( 'G loss: {} '.format(  l / steps  ) )
    print( 'Time: {} s'.format( time.time() - start_t ) )
    out =  G.predict( validation_noise ) * 255
    out = image_grid( out )
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
