import theano
import numpy as np
import sys
from six.moves import cPickle

'''
Author: Ross Briden, 09/08/16
Purpose: to unpickle theano paramaters into a type friendly, numpy format.
@params: argv0 = filename, argv1 = name of saved numpy array
'''

def _get_weights(weights):
    np.save(sys.argv[2] + '_weights.npy', weights.get_value())

def _get_biases(biases):
    np.save(sys.argv[2] + '_bias.npy', biases.get_value())

if __name__=="__main__":
    theano.config.experimental.unpickle_gpu_on_cpu = True
    pickled_file = open(sys.argv[1], 'rb')
    param_dict = cPickle.load(pickled_file)
    _get_weights(param_dict['W'])
    _get_biases(param_dict['b'])
