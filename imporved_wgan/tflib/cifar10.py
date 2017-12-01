import numpy as np

import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir, class_label=None, size=None):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if class_label is not None:
        timages = []
        tlabels = []
        for i, img in enumerate( images ):
            if labels[ i ] == class_label:
                # print(i)
                timages.append( img )
                tlabels.append( labels[ i ] )
        images = np.array(timages)
        labels = np.array(tlabels)

    if size is not None:
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        images = images[ : size ]
        labels = labels[ : size ]
    print( images.shape )

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir, class_label=None, size=None):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir, class_label, size),
        cifar_generator(['test_batch'], batch_size, data_dir, class_label, size)
    )


if __name__ == '__main__':
    res = load( 64, '../cifar10/cifar-10-batches-py' )
    d,t = res
    print(d)
    print(d())
    for i, l in d():
        print( l )
