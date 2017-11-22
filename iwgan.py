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


def multiply_mean( y_true, y_pred ):
    return K.mean( y_true * y_pred )

def mean( y_true, y_pred ):
    return K.mean( y_pred )


class IWGAN( object ):

    def __init__( self, discriminator, generator, opt=Adam(lr=0.0001, beta_1=0.5, beta_2=0.9), lmbd=10 ):
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
            batch_size=50,
            k=2,
            out_dir='./out',
            out_iter=5 ):
        #some helpful things
        ones = np.ones( ( batch_size, 1 ) )
        minus_ones = ones * -1.
        timer = Timer( iterations )
        output_pattern = out_dir + '/{:0' + str( len( str( iterations ) ) ) + 'd}.png' #erghh if it is stupid but it works it is not stupid
        progress = '{} | D( loss:\t{:0.2f}, diff:\t{:0.2f}, norm:\t{:0.2f}, ; G( loss:\t{:0.2f}  )'

        #get some noise:
        out_samples = self.make_some_noise()

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

            #trian generator
            self.make_trainable( False )
            g_loss = self.gan.train_on_batch( self.make_some_noise( batch_size ), minus_ones )

            #something messed up
            for l in self.gan.layers:
                weights = l.get_weights()
                replace = False
                for j, w in enumerate( weights ):
                    if np.isnan( w ).any():
                        print('\nfucking NaN man')
                        weights[ l ] = np.nan_to_num( w )
                        replace = True
                if replace:
                    l.set_weights( weights )

            if i % out_iter == 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # print(  self.generate( out_samples )  )
                    # print(  np.clip( ( self.generate( out_samples ) + 1 ) * 127.5, 0, 255 ) )
                    io.imsave( output_pattern.format( i ),
                        image_grid( np.clip( ( self.generate( out_samples ) + 1 ) * 127.5, 0, 255 ) ), # make sure we have valid values
                        plugin='pil' )




            timer.stop_step()
            #progess reporting
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
    (D, G, data) = zoo.cifar10_0()
    gan = IWGAN( D, G )
    gan.fit(data, iterations=10000,
                out_iter=50 )
