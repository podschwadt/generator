from __future__ import absolute_import
from sprites import dataset
from sprites import tools

import os, sys
sys.path.append(os.getcwd())
import os.path

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot
import cPickle as pickle



MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 20000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)
PENALTY = 100.
SEED = 7

lib.print_model_settings(locals().copy())

output_pattern = '{:0' + str( len( str( ITERS ) ) ) + 'd}.png' #erghh if it is stupid but it works it is not stupid

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])




######################################################
#gen training\
######################################################
#loop over all fake_data
#for all f in fake_data calculate dif = training_set - f
#distance = mean(training_set - f, axis=1)
#if min( distance ) < threshold add penalty
# diff_lambda = lambda x: diff_function( x )


noise = tf.placeholder( tf.float32, shape=[BATCH_SIZE, 128] )
fake_data = Generator( BATCH_SIZE, noise=noise )
real_images = 2*((tf.cast(tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM]), tf.float32)/255.)-.5)

def diff_function( x ):
    return tf.reduce_mean( tf.squared_difference( real_images, x ), axis=0 )

diff_sums = tf.map_fn( diff_function, fake_data )
penalty = tf.placeholder( tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

disc_fake = Discriminator(fake_data)

_pen = tf.constant( np.ones( ( BATCH_SIZE, OUTPUT_DIM ) ).astype('float32') * PENALTY )

gen_cost = - tf.reduce_mean(disc_fake) + tf.multiply( _pen, penalty )




######################################################
#disc training\
######################################################
fake_data_dt = Generator( BATCH_SIZE )
fake_data_disc = Discriminator( fake_data_dt )

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
disc_real = Discriminator(real_data)


disc_cost = tf.reduce_mean(fake_data_disc) - tf.reduce_mean(disc_real)

# Gradient penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1],
    minval=0.,
    maxval=1.
)
differences = fake_data_dt - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty




gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')
gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)



# For generating samples

if os.path.isfile( 'noise_file.dat' ):
    loaded_noise = pickle.load( open( 'noise_file.dat', 'rb' ) )
else:
    loaded_noise = np.random.normal(size=(128, 128)).astype('float32')
    pickle.dump( loaded_noise, open( 'noise_file.dat' , 'wb' ) )



fixed_noise_128 = tf.constant( loaded_noise )
fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), output_pattern.format(frame))

# For calculating inception score
samples_100 = Generator(100)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators

( x_train, _ ), ( x_test, _ ) = dataset.load( 'sprites/dataset_32x32.data', channels_first=True, seed=SEED )

x_train = x_train.reshape( x_train.shape[ 0 ], -1 )
x_test = x_test.reshape( x_test.shape[ 0 ], -1 )

all_images = np.concatenate( ( x_train, x_test ), axis=0 )

def inf_train_gen():
    while True:
        np.random.shuffle( x_train )
        yield x_train[ : BATCH_SIZE ]

def dev_gen():
    for i in xrange(len(x_test) / BATCH_SIZE):
        yield (x_test[i*BATCH_SIZE:(i+1)*BATCH_SIZE], 0 ) # second value is bogus and never used


#print some real samples
some_realsamples = []
lib.save_images.save_images(x_train[ : 128 ].reshape((128, 3, 32, 32)), 'real_samples.png')

# Train loop
timer = tools.Timer(ITERS)
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        timer.start_step()
        start_time = time.time()

        #first things first create fake_data


        _noise =  np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')

        # Train generator
        if iteration > 0:
            _diffs = None
            #calculate penalty:
            np.random.shuffle( all_images )
            for i in xrange( ( len( all_images ) / BATCH_SIZE ) -1 ):
                if _diffs is None:
                    _diffs = session.run( diff_sums, feed_dict={ noise: _noise, real_images: all_images[ i : i+1 ] } )
                else:
                    _diffs = np.minimum( session.run( diff_sums, feed_dict={ noise: _noise, real_images: all_images[ i : i+1 ] } ), _diffs )
            print( _diffs )

            _ = session.run( gen_train_op, feed_dict={ noise: _noise, penalty: _diffs } )
        # Train critic
        for i in xrange( CRITIC_ITERS ):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})


        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate inception score every 1K iters
        if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
        timer.stop_step()
        timer.out()
