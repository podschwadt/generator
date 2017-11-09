import numpy as np

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
        return ( sum_it() * 1. ) / len( self.list )

    def sum_it( self ):
        return sum( self.list )

    def variance( self ):
        return np.var( self.list )
