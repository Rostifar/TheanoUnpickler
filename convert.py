import theano
import numpy as np
from six.moves import cPickle
import os
import tensorflow as tf

'''
Author: Ross Briden, 09/09/16
purpose: simple module which takes a filename as input and unpickles the theano
parameters into numpy arrays
@returns: parameter directory
'''

def unpickle_parameters(filename):
    theano.config.experimental.unpickle_gpu_on_cpu = True
    pickled_file = open(filename, 'rb')
    param_dictionary = cPickle.load(pickled_file)
    print(param_dictionary)
    return param_dictionary

def get_scale1_variables():
    scale1_params = ['params-imnet_conv1_1.pk', 'params-imnet_conv1_2.pk', 'params-imnet_conv2_1.pk',
    'params-imnet_conv2_2.pk', 'params-imnet_conv3_1.pk', 'params-imnet_conv3_2.pk', 'params-imnet_conv3_3.pk',
    'params-imnet_conv4_1.pk', 'params-imnet_conv4_2.pk', 'params-imnet_conv4_3.pk', 'params-imnet_conv5_1.pk', 
    'params-imnet_conv5_2.pk', 'params-imnet_conv5_3.pk']
    variables = []
    with tf.variable_scope("scale1") as scope:
        for file in scale1_params:
            indx = str(scale1_params.index(file))
            mapping = {0:"W"+indx, 1:"b"+indx}
            tensor = convert_parameters_to_tensors(unpickle_parameters(file))
            for obj in tensor:
                tensor[tensor.index(obj)] = tf.Variable(obj, name=mapping[tensor.index(obj)]+indx)
            variables.append(tensor)
        return variables

def get_scale2_tensor():
    scale2_params = ['params-conv_s2_1.pk', 'params-depths_conv_s2_2.pk', 'params-depths_conv_s2_3.pk',
    'params-depths_conv_s2_4.pk', 'params-depths_conv_s2_5.pk']
    variables = []
    with tf.variable_scope("scope2") as scope:
        for file in scale2_params:
            indx = str(scale2_params)
            mapping = {0: "W"+indx, 1: "b"+indx}
            tensor = convert_parameters_to_tensors(unpickle_parameters(file))
            for obj in tensor:
                tensor[tensor.index(obj)] = tf.Variable(obj, name=mapping[tensor.index(obj)]+indx)
            variables.append(tensor)
        return variables

def get_scale3_tensor():
    scale3_params = ['params-conv_s3_1.pk', 'params-depths_conv_s3_2.pk', 'params-depths_conv_s3_3.pk',
    'params-depths_conv_s3_4.pk']
    variables = []
    for file in scale3_params:
        indx = str(scale3_params)
        mapping = {0: "W"+indx, 1: "b"+indx}
        tensor = convert_parameters_to_tensors(unpickle_parameters(file))
        for obj in tensor:
            tensor[tensor.index(obj)] = tf.Variable(obj, name=mapping[tensor.index(obj)]+indx)
        variables.append(tensor)
    return variables

def unpickle_all_paramters_in_model(dir):
    '''
    Input: desired directory
    @returns: all_parameters, a list of all the unpickled files found in
    the inputed directory; points to a dict['W': W, 'params':[W, b], 'b':b] 
    for each item in the list
    '''
    all_parameters = []
    for file in os.listdir(dir):
        if file.endswith(".pk"):
            all_parameters.append(unpickle_parameters(file))
    return all_parameters

def _get_tensor_value(np_array):
    '''
    converts a numpy array to a tensorflow tensor
    '''
    return tf.convert_to_tensor(np_array, dtype=tf.float32)

def convert_parameters_to_tensors(list_of_parameters):
    '''
    Inputs: list of parameters or single value
    @returns: a list of tensors
    '''
    if type(list_of_parameters) == type([]):
        for params_dict in list_of_parameters:
            index = list_of_parameters.index(params_dict)
            val = [_get_tensor_value(params_dict['W'].get_value()), 
            _get_tensor_value(params_dict['b'],get_value())]  
            list_of_parameters.insert(index, val)
        return list_of_parameters
    
    else: 
        try: 
            return [_get_tensor_value(list_of_parameters['W'].get_value()), 
        _get_tensor_value(list_of_parameters['b'].get_value())]

        except KeyError as e:
            return [_get_tensor_value(list_of_parameters['W'].get_value()), 
        _get_tensor_value(list_of_parameters['bias'].get_value())]