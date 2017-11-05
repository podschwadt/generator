from glob import glob
from PIL import Image

files = glob( 'images/*.png' )

xmax = 0
ymax = 0

for f in files:
    image = Image.open( f )
    ymax = max( image.height, ymax )
    xmax = max( image.width, xmax )
print( ymax )
print( xmax )

new_size = ( xmax, ymax )
bg_color = ( 0, 128, 128 )
folder = 'clean_images'
for f in files:
    print( f )
    old_im = Image.open( f )
    old_size = old_im.size
    new_im = Image.new( 'RGB', new_size, color = bg_color )
    new_im.paste( old_im, ( ( new_size[ 0 ] - old_size[ 0 ] ) / 2,( new_size[ 1 ] - old_size[ 1 ]  ) / 2 ) )
    new_im.save( folder + f.split( '/' )[ 1 ] )
