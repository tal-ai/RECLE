from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

def mnist_model(input, l1_n, l2_n, l3_n, reg_scale, reuse=False):
	with tf.name_scope("model"):
		with tf.variable_scope("layer1") as scope:
			net = tf.contrib.layers.fully_connected(input, l1_n, weights_regularizer = layers.l2_regularizer(reg_scale), \
																	activation_fn = tf.nn.sigmoid, scope=scope, reuse=reuse)
			reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'layer1')

		with tf.variable_scope("layer2") as scope:
			net = tf.contrib.layers.fully_connected(net, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale), \
																		activation_fn = tf.nn.sigmoid, scope=scope, reuse=reuse)
			reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'layer2')

		with tf.variable_scope("layer3") as scope:
			net = tf.contrib.layers.fully_connected(net, l3_n, weights_regularizer = layers.l2_regularizer(reg_scale), \
																				activation_fn=None, scope=scope, reuse=reuse)
			reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'layer3')
	
	return net

def triplet_loss(a, p, n, margin):
	with tf.name_scope("triplet-loss"):
		d_a_p = tf.reduce_sum(tf.square(a - p), 1)
		d_a_n = tf.reduce_sum(tf.square(a - n), 1)

		loss = tf.reduce_sum(tf.maximum((d_a_p - d_a_n + margin), 0))

		var_lst = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss = 0
		# for var in var_lst:
		# 	reg_loss += tf.reduce_sum(var)
		# reg_loss = tf.cast(reg_loss, dtype=tf.float64)
		
		total_loss = (reg_loss + loss) / tf.cast(tf.shape(a)[0], tf.float64)

		return total_loss, reg_loss