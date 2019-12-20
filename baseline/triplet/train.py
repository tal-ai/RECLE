from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from dataset import Dataset
from model import *
tf.logging.set_verbosity(tf.logging.ERROR)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_val_pairs', int(1e4), 'Validation pair size. ')
flags.DEFINE_integer('train_iter', 500, 'Total training iter')
flags.DEFINE_string('which_data', 'xueqian', 'specify which dataset to run')



def get_dummy_pairs(features):
	# anchor, positive and negative in dummy pairs are identical,
	# only to fit the architecture of model so that we can 
	# apply network on raw features and get embeddings
	n = features.shape[0]
	feature_a = [x for x in features]
	feature_p = feature_a
	feature_n = feature_a
	return feature_a, feature_p, feature_n


def apply_network_get_embd(raw_features, embd_save_path):
	# apply network and get robust embedding for training, validation and test data
	if('.csv' in embd_save_path):
		embd_save_path = embd_save_path.split('.csv')[0]
	print('Apply neural network to get robust embedding...')
	tmp_pairs = get_dummy_pairs(raw_features)
	tmp_a, tmp_p, tmp_n = tmp_pairs
	_anchor_embd, _pos_embd, _neg_embd = sess.run([anchor_embd, pos_embd, neg_embd], feed_dict={anchor:tmp_a, pos:tmp_p, neg: tmp_n})
	pd.DataFrame(_anchor_embd).to_csv(embd_save_path+'_anchor.csv', index=False)
	pd.DataFrame(_pos_embd).to_csv(embd_save_path+'_pos.csv', index=False)
	pd.DataFrame(_neg_embd).to_csv(embd_save_path+'_neg.csv', index=False)


if __name__ == "__main__":
	#setup dataset
	which_data = FLAGS.which_data
	train = pd.read_csv('/workspace/Guowei/paper/TKDE/data/{}/train.csv'.format(which_data))
	features_tr = np.array(train)[:, 2:]
	feature_dim = features_tr.shape[-1]
	labels_tr = np.array(train)[:, 0]
	labels_tr = np.array(labels_tr, dtype=np.float64)
	dataset_tr = Dataset(features_tr, labels_tr)

	val = pd.read_csv('/workspace/Guowei/paper/TKDE/data/{}/validation.csv'.format(which_data))
	features_val = np.array(val)[:, 2:]
	labels_val = np.array(val)[:, 0]
	labels_val = np.array(labels_val, dtype=np.float64)
	dataset_val = Dataset(features_val, labels_val)

	test = pd.read_csv('/workspace/Guowei/paper/TKDE/data/{}/test.csv'.format(which_data))
	features_ts = np.array(test)[:, 2:]
	labels_ts = np.array(test)[:, 0]
	labels_ts = np.array(labels_ts, dtype=np.float64)
	dataset_ts = Dataset(features_ts, labels_ts)

	l1_n_lst = [128, 64] + [1024, 512]
	l2_n_lst = [64, 32] + [512, 256, 128]
	l3_n_lst = [32, 16] + [128, 64]
	batch_size_lst = [256, 512]
	lr_lst = [0.005, 0.0005]
	reg_scale_lst = [2.0, 5.0]
	margin_lst = [0.1, 0.5, 1.0, 10.0]

	for _ in range(200):
		l1_n = random.choice(l1_n_lst)
		l2_n = random.choice(l2_n_lst)
		l3_n = random.choice(l3_n_lst)
		batch_size = random.choice(batch_size_lst)
		lr = random.choice(lr_lst)
		reg_scale = random.choice(reg_scale_lst)
		margin = random.choice(margin_lst)

		if(not (l1_n>=l2_n and l2_n>=l3_n)):
			continue
		model_name = 'l1_{}_l2_{}_l3_{}_bs_{}_lr_{}_margin_{}'.format(l1_n, l2_n, l3_n, batch_size, lr, margin)
		model_save_path = '/workspace/Guowei/paper/TKDE/{}/triplet/model/{}'.format(which_data,	model_name)
		embd_save_path = '/workspace/Guowei/paper/TKDE/{}/triplet/embd/{}'.format(which_data, model_name)

		if(not os.path.exists(model_save_path)):
			os.makedirs(model_save_path)
		else:
			print('pass model {}'.format(model_name))
			continue
		if(not os.path.exists(embd_save_path)):
			os.makedirs(embd_save_path)

		tf.reset_default_graph()
		model = mnist_model
		placeholder_shape = [None] + list(np.array(dataset_tr.pos_features).shape[1:]) 

		# Setup network
		next_batch_tr = dataset_tr.get_triplet_batch
		validation_pairs = dataset_val.get_triplet_batch(FLAGS.num_val_pairs)

		anchor = tf.placeholder(tf.float64, placeholder_shape, name='anchor')
		pos = tf.placeholder(tf.float64, placeholder_shape, name='pos')
		neg = tf.placeholder(tf.float64, placeholder_shape, name='neg')

		
		anchor_embd = model(anchor, l1_n, l2_n, l3_n, reg_scale, reuse=False)
		pos_embd = model(pos, l1_n, l2_n, l3_n, reg_scale, reuse=True)
		neg_embd = model(neg, l1_n, l2_n, l3_n, reg_scale, reuse=True)

		loss, reg_loss = triplet_loss(anchor_embd, pos_embd, neg_embd, margin)

		# Setup Optimizer
		global_step = tf.Variable(0, trainable=False)
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		gvs = optimizer.compute_gradients(loss)
		capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
		train_step = optimizer.apply_gradients(capped_gvs)

		# Start Training
		best_val_loss = 1e9
		previous_val_loss = 1e9
		earlyStopCount = 0
		MAX_EARLY_STOP_COUNTS = 5
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			#setup tensorboard	
			tf.summary.scalar('step', global_step)
			tf.summary.scalar('loss', loss)
			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter('train.log', sess.graph)

			#train iter
			for i in range(FLAGS.train_iter):
				# try:
				batch_a, batch_p, batch_n = next_batch_tr(batch_size)

				_, l = sess.run([train_step, loss], feed_dict={anchor:batch_a, pos:batch_p, neg: batch_n})
				print("\r#%d - Train loss"%i, l)
				if(i%2==0):
					batch_a, batch_p, batch_n = validation_pairs
					val_loss = sess.run(loss, feed_dict={anchor:batch_a, pos:batch_p, neg: batch_n})
					print('*'*66)
					print('epoch {} validation loss is {}'.format(i, val_loss))
					print('*'*66)
					
					if(val_loss<best_val_loss):
						best_val_loss = val_loss
						print('Epoch {} best model saved!'.format(i))
						saver.save(sess, "{}/model.ckpt".format(model_save_path))

						apply_network_get_embd(features_tr, os.path.join(embd_save_path, 'train_embd'))
						apply_network_get_embd(features_val, os.path.join(embd_save_path, 'val_embd'))
						apply_network_get_embd(features_ts, os.path.join(embd_save_path, 'test_embd'))

					if(val_loss>previous_val_loss):
						earlyStopCount += 1
					else:
						earlyStopCount = 0
					previous_val_loss = val_loss 
					print('current early stop count {}, max count {}'.format(earlyStopCount, MAX_EARLY_STOP_COUNTS))
					print('current best validation loss {}'.format(best_val_loss))

					if(earlyStopCount>=MAX_EARLY_STOP_COUNTS):
						print('Early stop at epoch {}!'.format(i))
						break





