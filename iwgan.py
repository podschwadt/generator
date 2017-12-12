from layers import Subtract, GradNorm
from tools import Timer, image_grid
import numpy as np
import zoo
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input
from keras.models import Sequential, Model
import keras.backend as K
import warnings
from skimage import io
import sys
import scipy.spatial.distance as sp_dist


# np.set_printoptions( threshold=np.nan, linewidth=np.nan, precision=2  )

def multiply_mean( y_true, y_pred ):
    return K.mean( y_true * y_pred )

def mean( y_true, y_pred ):
    return K.mean( y_pred )

def distance( x ):
    if len( x.shape ) == 4:
        x = np.mean( x, 3 )
    result = np.zeros( ( x.shape[ 0 ], x.shape[ 0 ] ) )
    for i in range( x.shape[ 0 ] ):
        for j in range( x.shape[ 0 ] ):
            result[ i ][ j ] = np.mean( ( x[ i ] - x[ j ] ) ** 2 )
    return result


class IWGAN( object ):

    def __init__( self, discriminator, generator,
                opt=Adam(
                    lr=0.0001,
                    beta_1=0.5,
                    beta_2=0.9,
                    # decay=1e-7
                    )
                ,lmbd=10 ):
        self.generator = generator
        self.discriminator = discriminator

        self.make_trainable( False )
        self.gan = Sequential([generator, discriminator])
        self.gan.compile(optimizer=opt, loss=multiply_mean)

        self.make_trainable( True )
        shape = self.discriminator.get_input_shape_at( 0 )[ 1: ]
        gen_input, real_input, interpolation = Input( shape ), Input( shape ), Input( shape )
        sub = Subtract()( [ self.discriminator( gen_input ), self.discriminator( real_input ) ] )
        norm = GradNorm()( [ self.discriminator( interpolation ), interpolation ] )
        self.dis_trainer = Model( [ gen_input, real_input, interpolation ], [ sub, norm ] )

        self.dis_trainer.compile( optimizer=opt, loss=[ mean, 'mse' ], loss_weights=[ 1.0, lmbd ] )

    def fit( self, data,
            iterations=1000,
            batch_size=64,
            k=2,
            out_dir='./out',
            out_iter=5 ):
        #some helpful things
        ones = np.ones( ( batch_size, 1 ) ).astype( 'float32' )
        minus_ones = ones * -1.
        timer = Timer( iterations )
        output_pattern = out_dir + '/{:0' + str( len( str( iterations ) ) ) + 'd}.png' #erghh if it is stupid but it works it is not stupid
        clear = '\r                                                                                                                                                '
        progress = '{} | D( loss:\t{:0.2f}, diff:\t{:0.2f}, norm:\t{:0.2f}, ; G( loss:\t{:0.2f}  )'

        #get some noise:
        out_samples = self.make_some_noise()
        distance_samples = self.make_some_noise( 10 )

        print( distance( distance_samples ) )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # print( distance( np.clip( ( self.generate( distance_samples ) + 1 ) * 127.5, 0, 255 ) ) )
            io.imsave( out_dir + '/real_samples.png',
                image_grid( np.clip( ( data[ : 64 ] + 1 ) * 127.5, 0, 255 ) ), # make sure we have valid values
                plugin='pil' )

        for i in range( iterations ):
            timer.start_step()
            #train discriminator
            self.make_trainable( True )
            for j in range( k ):
                real_data = data[ np.random.choice( data.shape[0], batch_size, replace=False ), : ]
                fake_data = self.generate( self.make_some_noise( batch_size ) )
                epsilon = np.random.random( batch_size )
                interpolation = ( real_data.T * epsilon ).T + ( fake_data.T * ( 1 - epsilon ) ).T
                d_loss, d_diff, d_norm = self.dis_trainer.train_on_batch( [ real_data, fake_data, interpolation ], [ ones ] * 2 )

                ##something messed up
                # for l in self.dis_trainer.layers:
                #     weights = l.get_weights()
                #     replace = False
                #     for j, w in enumerate( weights ):
                #         if np.isnan( w ).any():
                #             weights[ j ] = np.nan_to_num( w )
                #             replace = True
                #     if replace:
                #         l.set_weights( weights )
                # if replace:
                #     print('\nfucking NaN man')

            #trian generator
            self.make_trainable( False )
            g_loss = self.gan.train_on_batch( self.make_some_noise( batch_size ), minus_ones )

            ##something messed up
            # for l in self.gan.layers:
            #     weights = l.get_weights()
            #     replace = False
            #     for j, w in enumerate( weights ):
            #         if np.isnan( w ).any():
            #             weights[ j ] = np.nan_to_num( w )
            #             replace = True
            #     if replace:
            #         l.set_weights( weights )
            # if replace:
            #     print('\nfucking NaN man')

            if np.isnan( d_loss ):
                for j,l in enumerate( self.gan.layers):
                    for k, w in enumerate(  l.get_weights() ):
                        w = np.nan_to_num( w )
                        print( '{}/{}: {}/{}'.format( j, k, np.min( w ), np.max( w ) )  )


            if i % out_iter == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # print( distance( np.clip( ( self.generate( distance_samples ) + 1 ) * 127.5, 0, 255 ) ) )
                    io.imsave( output_pattern.format( i ),
                        image_grid( np.clip( ( self.generate( out_samples ) + 1 ) * 127.5, 0, 255 ) ), # make sure we have valid values
                        plugin='pil' )




            timer.stop_step()
            #progess reporting
            sys.stdout.write( clear )
            sys.stdout.flush()
            sys.stdout.write( '\r' + progress.format( timer.out_str(), d_loss, d_diff, d_norm, g_loss ) )
            sys.stdout.flush()


    def generate( self, inputs ):
        return self.generator.predict( inputs )

    def make_some_noise( self, n=100 ):
        return np.random.uniform( -1., 1., size=( n, self.generator.get_input_shape_at( 0 )[ 1 ] ) ).astype('float32')

    def make_trainable( self, trainable=True ):
        self.discriminator.trainable = trainable
        for l in self.discriminator.layers:
            l.trainable = trainable

if __name__ == '__main__':
    (D, G, data) = zoo.cifar10_iwgan_paper( filters=64 )
    gan = IWGAN( D, G )
    gan.fit(data, iterations=5000,
                out_iter=100 )
