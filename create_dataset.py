from scipy.ndimage import imread
from glob import glob
import numpy as np
import cPickle as pickle


outfile = 'dataset.data'
files = glob( 'clean_images/*.png' )

result = None

for f in files:
    print( f )
    im = imread( f )
    if result is None:
        result = np.array( [ im ] )
    else:
        result = np.append( result, [ im ], axis = 0 )

print( result )
print( result.shape )

pickle.dump( result, open( outfile , 'wb' ) )
