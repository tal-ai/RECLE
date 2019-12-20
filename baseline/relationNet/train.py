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
	# query, sample1 and sample2 in dummy pairs are identical,
	# only to fit the architecture of model so that we can 
	# apply network on raw features and get embeddings
	n = features.shape[0]
	feature_q = [x for x in features]
	feature_s1 = feature_q
	feature_s2 = feature_q
	label = np.ones((2*n, 2), dtype=np.float64)
	return feature_q, feature_s1, feature_s2, label


def apply_network_get_embd(raw_features, embd_save_path):
	# apply network and get robust embedding for training, validation and test data
	if('.csv' in embd_save_path):
		embd_save_path = embd_save_path.split('.csv')[0]
	print('Apply neural network to get robust embedding...')
	tmp_pairs = get_dummy_pairs(raw_features)
	tmp_q, tmp_s1, tmp_s2, tmp_label = tmp_pairs
	embd = sess.run(q_embd, feed_dict={query : tmp_q, 
										sample1 : tmp_s1, 
										sample2 : tmp_s2, 
										y: tmp_label
									})
	pd.DataFrame(embd).to_csv(embd_save_path+'.csv', index=False)


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

	embd_layer_l1_n_lst = [128, 64]
	embd_layer_l2_n_lst = [64, 32] 
	embd_layer_l3_n_lst = [32, 16]
	relation_layer_l1_n_lst = [64, 32]
	relation_layer_l2_n_lst = [32, 16]
	batch_size_lst = [256, 512]
	lr_lst = [0.005, 0.0005]

	for _ in range(200):
		embd_l1_n = random.choice(embd_layer_l1_n_lst)
		embd_l2_n = random.choice(embd_layer_l2_n_lst)
		embd_l3_n = random.choice(embd_layer_l3_n_lst)

		rel_l1_n = random.choice(relation_layer_l1_n_lst)
		rel_l2_n = random.choice(relation_layer_l2_n_lst)

		batch_size = random.choice(batch_size_lst)
		lr = random.choice(lr_lst)

		if(not (embd_l1_n>=embd_l2_n and embd_l2_n>=embd_l3_n and rel_l1_n>=rel_l2_n)):
			continue
		model_name = 'embd_l1_{}_l2_{}_l3_{}_rel_l1_{}_l2_{}_bs_{}_lr_{}'.format(embd_l1_n, embd_l2_n, embd_l3_n, rel_l1_n, rel_l2_n, batch_size, lr)
		model_save_path = '/workspace/Guowei/paper/TKDE/{}/relation/model/{}'.format(which_data, model_name)
		embd_save_path = '/workspace/Guowei/paper/TKDE/{}/relation/embd/{}'.format(which_data, model_name)

		if(not os.path.exists(model_save_path)):
			os.makedirs(model_save_path)
		else:
			print('pass model {}'.format(model_name))
			continue
		if(not os.path.exists(embd_save_path)):
			os.makedirs(embd_save_path)

		tf.reset_default_graph()
		embd_model = embedding_module
		relation_model = relation_module

		placeholder_shape = [None] + list(np.array(dataset_tr.pos_features).shape[1:]) 

		# Setup network
		next_batch_tr = dataset_tr.get_triplet_batch
		validation_pairs = dataset_val.get_triplet_batch(FLAGS.num_val_pairs)

		query = tf.placeholder(tf.float64, placeholder_shape, name='query')
		sample1 = tf.placeholder(tf.float64, placeholder_shape, name='sample_0')
		sample2 = tf.placeholder(tf.float64, placeholder_shape, name='sample_1')
		y = tf.placeholder(tf.float64, [None, 2], name='labels')

		
		q_embd = embd_model(query, embd_l1_n, embd_l2_n, embd_l2_n, reuse=False)
		s1_embd = embd_model(sample1, embd_l1_n, embd_l2_n, embd_l2_n, reuse=True)
		s2_embd = embd_model(sample2, embd_l1_n, embd_l2_n, embd_l2_n, reuse=True)

		q_s1_embd = tf.concat([q_embd, s1_embd], axis=1)
		q_s2_embd = tf.concat([q_embd, s2_embd], axis=1)

		logits1 = relation_model(q_s1_embd, rel_l1_n, rel_l2_n, reuse=False)
		logits2 = relation_model(q_s2_embd, rel_l1_n, rel_l2_n, reuse=True)
		logits = tf.concat([logits1, logits2], axis=0)

		loss = relation_loss(logits, y)

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
				batch_q, batch_s1, batch_s2, batch_y = next_batch_tr(batch_size)

				_, l = sess.run([train_step, loss], feed_dict={
																query : batch_q, 
																sample1 : batch_s1, 
																sample2 : batch_s2, 
																y: batch_y
																})
				print("\r#%d - Train loss"%i, l)
				if(i%2==0):
					valid_q, valid_s1, valid_s2, valid_y = validation_pairs
					val_loss = sess.run(loss, feed_dict={
														query : valid_q, 
														sample1 : valid_s1, 
														sample2 : valid_s2, 
														y: valid_y
														})
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





