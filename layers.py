from keras.layers.merge import _Merge
from keras.engine.topology import Layer
from keras import backend as K

# based on
# http://shaofanlai.com/post/10
# https://github.com/PiscesDream/Keras-GAN
class Subtract( _Merge ):
    def _merge_function( self, inputs ):
        output = inputs[ 0 ]
        for i in range( 1, len( inputs ) ):
            output = output-inputs[ i ]
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
