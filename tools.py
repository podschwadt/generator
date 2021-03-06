import numpy as np
from PIL import Image
import sys
import time



class CircleList( object ):

    def __init__( self, size ):
        self.size = size
        self.list = []
        self.index = 0

    def add( self, x ):
        if( len( self.list ) < self.size ):
            self.list.append( x )
        else:
            self.list[ self.index ] = x
        self.index = ( self.index + 1 ) % self.size

    def mean( self ):
        return ( self.sum_it() * 1. ) / len( self.list )

    def sum_it( self ):
        return sum( self.list )

    def variance( self ):
        return np.var( self.list )


class Timer( object ):

    def __init__( self, steps, size=100, print_steps=10 ):
        self.list = CircleList( 100 )
        self.steps = steps
        self.count = 0
        self.print_steps = print_steps

    def start_step( self ):
        self.start = time.time()

    def stop_step( self ):
        self.list.add( time.time() - self.start )
        self.count += 1

    def __str__( self ):
        return time.strftime( '%H:%M:%S', time.gmtime( ( self.steps - self.count ) * self.list.mean() ) )

    def out_str( self ):
        return 'remaining {} | {}/{} '.format( self, self.count, self.steps  )

    def out( self ):
        if self.count % self.print_steps == 0:
            sys.stdout.write( '\r' + self.out_str() )
            sys.stdout.flush()



def image_grid( images, bg_color=( 0, 0, 0 ), distance=( 0, 0 ), debug=False ):
    """
    images must be numpy array of images in int [0;255]
    """
    images = images.astype( 'uint8' )
    if images.shape[ 3 ] == 1:
        mode = 'L'
    elif images.shape[ 3 ] == 3:
        mode = 'RGB'
    else:
        raise Exception()
    y_max = 0
    x_max = 0
    for i in images:
        x_max = max( x_max, i.shape[ 0 ] )
        y_max = max( y_max, i.shape[ 1 ] )
    rows = np.ceil( np.sqrt( len( images ) ) ).astype( int )
    new_size = ( rows * x_max + rows * distance[ 0 ] , rows * y_max + rows * distance[ 1 ] )
    new_im = Image.new( 'RGB', new_size, color = bg_color )
    row = 0
    col = 0
    count = 1
    for i in images:
        if row == rows:
            col += 1
            row = 0

        if i.shape[ 2 ] == 1:
            i = i.reshape( i.shape[ : 2 ] )
        old_im = Image.fromarray( i, mode=mode )
        old_size = old_im.size
        new_im.paste( old_im, ( row * x_max + row * distance[ 0 ], col * y_max + col * distance[ 1 ]  ) )
        if debug:
            sys.stdout.write('\rProcessing image %i/%i' % ( count, len( images ) ) )
            sys.stdout.flush()
            count += 1
        row += 1

    return new_im
