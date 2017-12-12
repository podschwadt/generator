from scipy.ndimage import imread
from glob import glob
import numpy as np
import cPickle as pickle
import resize




_outfile = 'dataset.data'
files = glob( 'clean_images/*.png' )


def best_fit():
    resize.resize()
    out()

def custom_fit( dim ):
    resize.resize( dim )
    out( 'dataset_{}x{}.data'.format( dim[0], dim[1] ) )


def out( outfile ):
    result = None
    if outfile is None:
        outfile = _outfile
    for f in files:
        print( f )
        im = imread( f )
        if result is None:
            result = np.array( [ im ] )
        else:
            result = np.append( result, [ im ], axis = 0 )
            # print( result )
            # print( result.shape )

    pickle.dump( result, open( outfile , 'wb' ) )

if __name__ == '__main__':
    custom_fit( (32,32) )
