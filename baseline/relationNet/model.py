from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

def embedding_module(input, l1_n, l2_n, l3_n, reuse):
	with tf.name_scope("embedding_model"):
		with tf.variable_scope("embd_layer1") as scope:
			net = tf.contrib.layers.fully_connected(input, l1_n, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)

		with tf.variable_scope("embd_layer2") as scope:
			net = tf.contrib.layers.fully_connected(net, l2_n, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)

		with tf.variable_scope("embd_layer3") as scope:
			net = tf.contrib.layers.fully_connected(net, l3_n, activation_fn=None, scope=scope, reuse=reuse)
	return net


def relation_module(embd, l1_n, l2_n, reuse):
	with tf.name_scope('relation_model'):
		with tf.variable_scope('rel_layer1') as scope:
			net = tf.contrib.layers.fully_connected(embd, l1_n, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
		
		with tf.variable_scope("rel_layer2") as scope:
			net = tf.contrib.layers.fully_connected(net, l2_n, activation_fn=tf.nn.relu, scope=scope, reuse=reuse)
		
		with tf.variable_scope("rel_final_layer") as scope:
			logits = tf.contrib.layers.fully_connected(net, 2, activation_fn=tf.nn.sigmoid, scope=scope, reuse=reuse)
	return logits



def relation_loss(logits, y):
	with tf.name_scope("mse-loss"):
		loss = tf.reduce_sum(tf.pow(logits - y,2)) / tf.cast(tf.shape(logits)[0], tf.float64)
		return loss