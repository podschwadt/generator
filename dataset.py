import numpy as np
import cPickle as pickle

filename = 'dataset.data'

def load( ):
    x = pickle.load( open( filename, 'rb' ) )
    np.random.shuffle( x )
    y = np.arange( x.shape[ 0 ] )
    y = np.ones_like( y )
    y = y.astype( 'float32' )

    x_test = x[  : x.shape[ 0 ] / 10 ]
    x_train = x[ ( x.shape[ 0 ] / 10 ) + 1 :  ]
    y_test = y[  : ( x.shape[ 0 ] / 10 )  ]
    y_train = y[ ( x.shape[ 0 ] / 10 ) + 1 :  ]

    return ( x_train , y_train ), ( x_test , y_test )

if __name__ == '__main__':
    load()
