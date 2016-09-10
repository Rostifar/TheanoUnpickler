import theano
import numpy as np
from six.moves import cPickle

'''
Author: Ross Briden, 09/09/16
purpose: simple module which takes a filename as input and unpickles the theano
parameters into numpy arrays
@returns: weight numpy parameter, bias numpy parameter
'''

def unpickle_parameters(filename):
    theano.config.experimental.unpickle_gpu_on_cpu = True
    pickled_file = open(filename, 'rb')
    param_dictionary = cPickle.load(pickled_file)
    return param_dictionary['W'].get_value(), param_dictionary['b'].get_value()