from __future__ import absolute_import
from sprites import dataset
from sprites import tools

import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tflib as lib
import tflib.save_images

def get_penalty_matrix():
    #load data
    ( x_train, _ ), ( x_test, _ ) = dataset.load( 'sprites/dataset_32x32.data', channels_first=True )

    x_train = x_train.reshape( x_train.shape[ 0 ], -1 )
    x_test = x_test.reshape( x_test.shape[ 0 ], -1 )

    all_samples = np.concatenate( (x_train, x_test ), axis=0 ).astype('float32')

    print( all_samples.shape )

    result = []

    for idx, i in enumerate( all_samples ):
        for j in all_samples:
            result.append( np.abs( i - j ) )
    result = np.mean( result, axis=0 )

    return result
#lib.save_images.save_images(result.reshape((1, 3, 32, 32)), 'penalty.png')
