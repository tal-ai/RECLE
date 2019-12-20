from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
import numpy as np


class Dataset(object):
	def __init__(self, features, labels):
		self.pos_features = []
		self.neg_features = []
		for i, y in enumerate(labels):
			if(y==1):
				self.pos_features.append(np.array(features[i]))
			elif(y==0):
				self.neg_features.append(np.array(features[i]))
			else:
				raise ValueError('invalid label value={}, must be 0 or 1'.format(y))
		self.num_pos = len(self.pos_features)
		self.num_neg = len(self.neg_features)
		print('positive samples {} negative samples {}'.format(self.num_pos, self.num_neg))

	def _get_siamese_similar_pair(self):
		if(np.random.random() < 0.5):
			l_idx, r_idx = np.random.choice(self.num_pos, 2)
			l = self.pos_features[l_idx]
			r = self.pos_features[r_idx]
		else:
			l_idx, r_idx = np.random.choice(self.num_neg, 2)
			l = self.neg_features[l_idx]
			r = self.neg_features[r_idx]
		return l, r, 1

	def _get_siamese_dissimilar_pair(self):
		if(np.random.random() < 0.5):
			l_idx = np.random.choice(self.num_pos, 1)[0]
			l = self.pos_features[l_idx]
			r_idx = np.random.choice(self.num_neg, 1)[0]
			r = self.neg_features[r_idx]
		else:
			r_idx = np.random.choice(self.num_pos, 1)[0]
			r = self.pos_features[r_idx]
			l_idx = np.random.choice(self.num_neg, 1)[0]
			l = self.neg_features[l_idx]
		return l, r, 0

	def _get_siamese_pair(self):
		if np.random.random() < 0.5:
			return self._get_siamese_similar_pair()
		else:
			return self._get_siamese_dissimilar_pair()

	def get_siamese_batch(self, n):
		feature_left, feature_right, labels = [], [], []
		for _ in range(n):
			l, r, y = self._get_siamese_pair()
			feature_left.append(l)
			feature_right.append(r)
			labels.append(y)
		return feature_left, feature_right, np.array(labels).reshape(-1, 1)