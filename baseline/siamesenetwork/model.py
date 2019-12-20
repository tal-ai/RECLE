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

		with tf.variable_scope("layer2") as scope:
			net = tf.contrib.layers.fully_connected(net, l2_n, weights_regularizer = layers.l2_regularizer(reg_scale), \
																		activation_fn = tf.nn.sigmoid, scope=scope, reuse=reuse)

		with tf.variable_scope("layer3") as scope:
			net = tf.contrib.layers.fully_connected(net, l3_n, weights_regularizer = layers.l2_regularizer(reg_scale), \
																				activation_fn=None, scope=scope, reuse=reuse)
	return net

def contrastive_loss(model1, model2, y, margin):
	with tf.name_scope("contrastive-loss"):
		distance = tf.reduce_sum(tf.square(model1 - model2), 1)
		similarity = tf.reduce_sum(y * tf.square(distance))                                    # keep the similar label (1) close to each other
		dissimilarity = tf.reduce_sum((1 - y) * tf.square(tf.maximum((margin - distance), 0)))        # give penalty to dissimilar label if the distance is bigger than margin

		var_lst = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss = 0
		for var in var_lst:
			reg_loss += tf.reduce_sum(var)

		reg_loss = tf.cast(reg_loss, dtype=tf.float64)
		total_loss = (reg_loss + (dissimilarity + similarity) / 2.0) / tf.cast(tf.shape(model1)[0], tf.float64)

		return total_loss, reg_loss, similarity, dissimilarity