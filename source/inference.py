import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from utils import *
import warnings

# Given TF model path, return weight and bias of the neural net
def load_neuronet(model_path):
    restore_graph = tf.Graph()
    w, b ={}, {}
    with tf.Session(graph=restore_graph) as restore_sess:
        restore_saver = tf.train.import_meta_graph(os.path.join(model_path, 'rll.ckpt.meta'))
        restore_saver.restore(restore_sess, tf.train.latest_checkpoint(model_path))
        g = tf.get_default_graph()
        fc_l1_query_w = g.get_tensor_by_name('fc_l1_query/weights:0').eval()
        fc_l1_query_b = g.get_tensor_by_name('fc_l1_query/biases:0').eval()
        fc_l1_doc_w = g.get_tensor_by_name('fc_l1_doc/weights:0').eval()
        fc_l1_doc_b = g.get_tensor_by_name('fc_l1_doc/biases:0').eval()

        fc_l2_query_w = g.get_tensor_by_name('fc_l2_query/weights:0').eval()
        fc_l2_query_b = g.get_tensor_by_name('fc_l2_query/biases:0').eval()
        fc_l2_doc_w = g.get_tensor_by_name('fc_l2_doc/weights:0').eval()
        fc_l2_doc_b = g.get_tensor_by_name('fc_l2_doc/biases:0').eval()
        w['fc_l1_query_w'] = fc_l1_query_w
        w['fc_l1_doc_w'] = fc_l1_doc_w
        w['fc_l2_query_w'] = fc_l2_query_w
        w['fc_l2_doc_w'] = fc_l2_doc_w

        b['fc_l1_query_b'] = fc_l1_query_b
        b['fc_l1_doc_b'] = fc_l1_doc_b
        b['fc_l2_query_b'] = fc_l2_query_b
        b['fc_l2_doc_b'] = fc_l2_doc_b
        return w, b
    
# apply nonlinear activation 
def nonlinear(x, activation):
    x = np.array(x, dtype=np.float128)
    if(activation=='sigmoid'):
        return 1.0/(1+np.exp(-x))
    elif(activation=='tanh'):
        return np.tanh(x)
    elif(activation=='relu'):
        return x*(x>0)
    else:
        raise ValueError('activation must be in [sigmoid, tanh, relu.]')

# Obtain the learned embedding
def embeddingInference(w, b, x, which, activation):
    '''
    Args:
    w: neural net weight dictionary
    b: neural net bias dictionary
    x: raw feature in original space
    which: must be 'query' or 'doc' to specify which net to use. Using query is recommended
    activation: must be sigmoid, tanh or relu

    Returns:
    the learned embedding
    '''

    if(which=='query'):
        l1_out = nonlinear(np.dot(x, w['fc_l1_query_w'])+b['fc_l1_query_b'], activation)
        return nonlinear(np.dot(l1_out, w['fc_l2_query_w'])+b['fc_l2_query_b'], activation)
    elif(which =='doc'):
        l1_out = nonlinear(np.dot(x, w['fc_l1_doc_w'])+b['fc_l1_doc_b'], activation)
        return nonlinear(np.dot(l1_out, w['fc_l2_doc_w'])+b['fc_l2_doc_b'], activation)
    else:
        raise ValueError('which={} is invalid, must be query or doc'.format(which))

# Given TF model_path, a basic classifier like logistc regression classifier,
# and raw feature X, returns binary classification prediction y_hat and y_hat_proba
def inference(model_path, classifier, X):
    if('sigmoid' in model_path):
        activation = 'sigmoid'
    elif('tanh' in model_path):
        activation = 'tanh'
    elif('relu' in model_path):
        activation = 'relu'
    else:
        raise ValueError('invalid activation!')
    
    w, b = load_neuronet(model_path)
    embd = embeddingInference(w, b, X, 'doc', activation)
    y_hat = classifier.predict(embd)
    y_hat_proba = classifier.predict_proba(embd)
    return y_hat, y_hat_proba