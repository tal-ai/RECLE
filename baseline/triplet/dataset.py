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

	def _get_triplet_pair(self):
		if(np.random.random() < 0.5):
			anchor_idx = np.random.choice(self.num_pos)
			pos_idx = np.random.choice(self.num_pos)
			neg_idx = np.random.choice(self.num_neg)
			anchor = self.pos_features[anchor_idx]
			pos = self.pos_features[pos_idx]
			neg = self.neg_features[neg_idx]
		else:
			anchor_idx = np.random.choice(self.num_neg)
			pos_idx = np.random.choice(self.num_neg)
			neg_idx = np.random.choice(self.num_pos)
			anchor = self.neg_features[anchor_idx]
			pos = self.neg_features[pos_idx]
			neg = self.pos_features[neg_idx]
		return anchor, pos, neg

	def get_triplet_batch(self, n):
		feature_anchor, feature_pos, feature_neg = [], [], []
		for _ in range(n):
			anchor, pos, neg = self._get_triplet_pair()
			feature_anchor.append(anchor)
			feature_pos.append(pos)
			feature_neg.append(neg)
		return feature_anchor, feature_pos, feature_neg