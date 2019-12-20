import numpy as np
import pandas as pd
import os, time
import tensorflow as tf
import sys
import tensorflow.contrib.layers as layers
from inference import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score 
from utils import *
import warnings

warnings.filterwarnings('ignore')

'''
This code implements the RECLE framework
'''


'''
Parameters
dimension: feature dimension of raw data
bs: batch size
lr_rate: learning rate
l1_n: number of neurons in the first layer
l2_n: number of neurons in the second layer
max_iter: max interation number for training
reg_scale: regularization penalty
dropout_rate: ratio to drop neurons in each layer
gamma: a held-out hyperparameter in loss function, we set it to 10.0 in our experiments
save_path: where you save the model
model_name: you can name your model however you like
'''  
class RECLE(object):
    def __init__(self, dimension, l1_n, l2_n, gamma, distance_metric, train, validation):
        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
        self.train = train
        self.y_tr = train[:,0]
        self.X_tr = train[:,2:]
        self.y_val = validation[:,0]
        self.X_val = validation[:,2:]
        self.distance_metric = distance_metric
        self.current_train_step = 0
        self.current_val_step = 0


    def find_similar_sample(self, X, V_miss, distance_metric='cosine', top_k=3):
        '''
        compute distance of (v, x), for v in V_miss and x in X
        for each v in V_miss, find the cloeset top_k x in X
        args: 
        V_miss : a matrix of shape [num_examples, feature_dimension]
        '''
        target = []
        for v in V_miss:
            distance = compute_distance(X, v, distance_metric)
            target += [x for x in np.argsort(distance)[:top_k]]

        target = list(set(target)) # remove duplicates
        return self.train[target], target


    def feedBatch(self, groups, weights, batchSize, is_training):
        batchIndex = np.random.randint(low=0, high=groups[0].shape[0], size=batchSize)
        batchGroups = [groups[i][batchIndex] for i in range(len(groups))]
        batchWeights = [weights[i][batchIndex] for i in range(len(weights))]
        batchData = {
                    self.is_training: is_training,
                    self.query: batchGroups[0], 
                    self.posDoc : batchGroups[1], 
                    self.negDoc0 :batchGroups[2], 
                    self.negDoc1 : batchGroups[3], 
                    self.negDoc2: batchGroups[4],
                    self.posDocW: batchWeights[0].reshape(-1, ),    
                    self.negDoc0W: batchWeights[1].reshape(-1, ),
                    self.negDoc1W: batchWeights[2].reshape(-1, ),
                    self.negDoc2W: batchWeights[3].reshape(-1,)
                    }
        return batchData
    
    def buildRLL(self, lr_rate, reg_scale, dropout_rate, activation):
        tf.reset_default_graph()
        self.is_training = tf.placeholder_with_default(False, shape=(), name='isTraining')
        self.query = tf.placeholder(tf.float32, shape=[None, self.dimension], name='queryInput')
        self.posDoc = tf.placeholder(tf.float32, shape=[None, self.dimension], name='posDocInput')
        self.negDoc0 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc0Input')
        self.negDoc1 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc1Input')
        self.negDoc2 = tf.placeholder(tf.float32, shape=[None, self.dimension], name='negDoc2Input')
        self.posDocW = tf.placeholder(tf.float32, shape=[None], name='posDocWeight')
        self.negDoc0W = tf.placeholder(tf.float32, shape=[None], name='negDoc0Weight')
        self.negDoc1W = tf.placeholder(tf.float32, shape=[None], name='negDoc1Weight')
        self.negDoc2W = tf.placeholder(tf.float32, shape=[None], name='negDoc2Weight')
        if(activation=='sigmoid'):
            self.activation = tf.nn.sigmoid
        elif(activation=='relu'):
            self.activation = tf.nn.relu
        elif(activation=='tanh'):
            self.activation = tf.nn.tanh
        else:
            raise ValueError('activation function must be in ["sigmoid", "relu", "tanh"]')

        self.lr_rate = lr_rate
        self.reg_scale = reg_scale
        self.dropout_rate = dropout_rate

        with tf.name_scope('fc_l1_query'):
            outputQuery = tf.contrib.layers.fully_connected(self.query, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                 activation_fn = self.activation, scope='fc_l1_query')
            outputQuery = tf.layers.dropout(outputQuery, self.dropout_rate, training=self.is_training)
        with tf.name_scope('fc_l1_doc'):
            outputPosDoc =  tf.contrib.layers.fully_connected(self.posDoc, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                activation_fn = self.activation, scope='fc_l1_doc')
            outputPosDoc = tf.layers.dropout(outputPosDoc, self.dropout_rate, training=self.is_training)

            outputNegDoc0 =  tf.contrib.layers.fully_connected(self.negDoc0, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                 activation_fn = self.activation, scope='fc_l1_doc', reuse=True)
            outputNegDoc0 = tf.layers.dropout(outputNegDoc0, self.dropout_rate, training=self.is_training)

            outputNegDoc1 =  tf.contrib.layers.fully_connected(self.negDoc1, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                  activation_fn = self.activation, scope='fc_l1_doc', reuse=True)
            outputNegDoc1 = tf.layers.dropout(outputNegDoc1, self.dropout_rate, training=self.is_training)

            outputNegDoc2 =  tf.contrib.layers.fully_connected(self.negDoc2, self.l1_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                  activation_fn = self.activation, scope='fc_l1_doc', reuse=True)

            outputNegDoc2 = tf.layers.dropout(outputNegDoc2, self.dropout_rate, training=self.is_training)


        with tf.name_scope('fc_l2_query'):
            outputQuery = tf.contrib.layers.fully_connected(outputQuery, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                             activation_fn = self.activation, scope='fc_l2_query')
            outputQuery = tf.layers.dropout(outputQuery, self.dropout_rate, training=self.is_training)

        with tf.name_scope('fc_l2_doc'):
            outputPosDoc = tf.contrib.layers.fully_connected(outputPosDoc, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale), 
                                                               activation_fn = self.activation, scope='fc_l2_doc')
            outputPosDoc = tf.layers.dropout(outputPosDoc, self.dropout_rate, training=self.is_training)

            outputNegDoc0 = tf.contrib.layers.fully_connected(outputNegDoc0, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                activation_fn = self.activation, scope='fc_l2_doc', reuse=True)
            outputNegDoc0 = tf.layers.dropout(outputNegDoc0, self.dropout_rate, training=self.is_training)

            outputNegDoc1 = tf.contrib.layers.fully_connected(outputNegDoc1, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                activation_fn = self.activation, scope='fc_l2_doc', reuse=True)
            outputNegDoc1 = tf.layers.dropout(outputNegDoc1, self.dropout_rate, training=self.is_training)

            outputNegDoc2 = tf.contrib.layers.fully_connected(outputNegDoc2, self.l2_n, weights_regularizer = layers.l2_regularizer(self.reg_scale),\
                                                                activation_fn = self.activation, scope='fc_l2_doc', reuse=True)

            outputNegDoc2 = tf.layers.dropout(outputNegDoc2, self.dropout_rate, training=self.is_training)

        with tf.name_scope('loss'):
            reg_ws_0 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_query')
            reg_ws_1 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l1_doc')
            reg_ws_2 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_query')
            reg_ws_3 = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_l2_doc')
            reg_loss = tf.reduce_sum(reg_ws_0)+tf.reduce_sum(reg_ws_1)+tf.reduce_sum(reg_ws_2)+tf.reduce_sum(reg_ws_3)

            numerator = tf.multiply(self.posDocW, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputPosDoc))))
            doc0_similarity = tf.multiply(self.negDoc0W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc0))))
            doc1_similarity = tf.multiply(self.negDoc1W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc1))))
            doc2_similarity = tf.multiply(self.negDoc2W, tf.exp(tf.multiply(self.gamma, cos_sim(outputQuery, outputNegDoc2))))
            prob = numerator / tf.add(doc0_similarity+ doc1_similarity+doc2_similarity+numerator,tf.constant(1e-10))
            
            bs = tf.cast(tf.shape(self.query)[0], tf.float32)
            self.loss = -tf.reduce_sum(tf.log(prob)) / bs
            self.total_loss = self.loss + reg_loss / bs
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('log_and_reg_loss', self.total_loss)

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr_rate).minimize(self.total_loss)

        with tf.name_scope('Summary'):
            self.merged_summary_op = tf.summary.merge_all()

    def run_epoch(self, sess, batch_size, data, data_size, training, epoch, summary_writer):
        total_loss = 0
        num_batch = int(data_size / batch_size)
        groups, weights = data[0], data[1]
        for batch in range(num_batch):
            if(training):
                batchData = self.feedBatch(groups, weights, batch_size, is_training=True)
                _, batch_loss, summary = sess.run([self.optimizer, self.loss, self.merged_summary_op], feed_dict=batchData)
                summary_writer.add_summary(summary, self.current_train_step)
                self.current_train_step += 1
            else:
                batchData = self.feedBatch(groups, weights, batch_size, is_training=False)
                batch_loss, summary = sess.run([self.loss, self.merged_summary_op], feed_dict=batchData)
                summary_writer.add_summary(summary, self.current_val_step)
                self.current_val_step += 1
            total_loss += batch_loss
        
        total_loss /= num_batch
        if(training):
            print("Epoch {} train loss {}, train size {}".format(epoch, total_loss, data_size))
        else:
            print('*'*60)
            print("Epoch {} validation loss {}".format(epoch, total_loss))
            print('*'*60)
            print('\n')
        return total_loss

    def hard_example_mining(self, y_hat_val, space, groupsTr=None, weightsTr=None, polarity='positive'):
        '''
        space: in which space to find similar examples, must be ['raw', 'embedding']
        '''
        print('Perform hard example mining')
        # find mispredicted examples in validation set, e.g. V_miss
        mis_predicted = [i for i in range(self.y_val.shape[0]) if y_hat_val[i]!=self.y_val[i]]

        if(space=='raw'):
            V_miss = self.X_val[mis_predicted]
            # find most similar examples to each v in V_miss from and ONLY from the training set
            X_hard, hard_idx = self.find_similar_sample(self.X_tr, V_miss, self.distance_metric)
        elif(space=='embedding'):
            V_miss = self.embd_val[mis_predicted]
            X_hard, hard_idx = self.find_similar_sample(self.embd_tr, V_miss, self.distance_metric)
        else:
            raise ValueError('invalid space input={}, must be raw or embedding'.format(space))

        if(sum(self.y_tr[hard_idx])==0 or sum(self.y_tr[hard_idx])==len(hard_idx)):
            # in case of zero positive or zero negative examples in X_hard
            # just create hard groups randomly
            G_hard, G_hard_weight = prepareInput(self.train, groupSize=int(1e5), polarity=polarity)
        else:
            # create hard groups from X_hard, which are from training set
            G_hard, G_hard_weight = prepareInput(X_hard, groupSize=int(1e5), polarity=polarity)
        
        if(groupsTr==None or weightsTr==None):
            return (G_hard, G_hard_weight), G_hard[0].shape[0]
        else:
            # combine old training groups and G_hard
            new_groups, new_weights = [], []
            for base, hard in zip(groupsTr, G_hard):
                tmp = np.concatenate((base, hard), axis=0)
                new_groups.append(tmp)
            for base, hard in zip(weightsTr, G_hard_weight):
                tmp = np.concatenate((base, hard), axis=0)
                new_weights.append(tmp)
            new_tr_data = (new_groups, new_weights)
            new_tr_size = new_groups[0].shape[0]
            return new_tr_data, new_tr_size

    def train_and_evaluate(self, train, validation, batchSize, max_iter, 
                    activation, save_path, summaries_dir, model_name, MAX_EARLY_STOP_COUNTS=5, 
                    WARMUP_EPOCHS=5, WARMUP_ACC=0.7, tr_groups_size=int(4e5), val_groups_size=int(8e5), 
                    space='embedding', use_base_group=False, polarity='positive'):

        ratio = sum(train[:,0]) / train.shape[0]
        max_vote = max(train[:, 1])
        print('positive ratio {}, max vote {}'.format(ratio, max_vote))
        alpha = ratio*max_vote
        beta = max_vote - alpha
        
        # create initial training groups
        groupsTr, weightsTr = prepareInput(train, groupSize=tr_groups_size, alpha=alpha, beta=beta, polarity=polarity) 
        init_tr_data = (groupsTr, weightsTr)
        init_tr_size = groupsTr[0].shape[0]
        # create validation groups and don't change it
        groupsVal, weightsVal = prepareInput(validation, groupSize=val_groups_size, alpha=alpha, beta=beta, polarity=polarity) 
        val_data = (groupsVal, weightsVal)
        val_size = groupsVal[0].shape[0]

        best_val_loss = sys.maxsize
        best_acc = 0
        previous_val_loss = sys.maxsize
        earlyStopCount = 0
        saver = tf.train.Saver()
        print('training model {}'.format(model_name))

        currentModelPath = os.path.join(save_path, model_name)
        if(not os.path.exists(currentModelPath)):
            os.makedirs(currentModelPath)
            os.makedirs(currentModelPath+'/current')
            os.makedirs(currentModelPath+'/best_acc_model')

        summaries_dir = os.path.join(summaries_dir, model_name)
        if(not os.path.exists(summaries_dir)):
            os.makedirs(summaries_dir)

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(summaries_dir + '/validation', sess.graph)
            tf.global_variables_initializer().run()
            acc, prec, recall, auc = 0, 0, 0, 0
            for epoch in range(0, max_iter):
                if(epoch<WARMUP_EPOCHS):
                    # let model do some warmup training
                    self.run_epoch(sess, batchSize, init_tr_data, init_tr_size, True, epoch, train_writer)
                else:
                    # perform adaptive hard example mining
                    # forward prop to get embeddings, e.g. embd_tr and embd_val
                    w, b = load_neuronet(os.path.join(currentModelPath, 'current'))
                    self.embd_tr = embeddingInference(w, b, self.X_tr, 'doc', activation)
                    self.embd_val = embeddingInference(w, b, self.X_val, 'doc', activation)

                    # train classification model on (embd_tr, y_tr), predict on validation set
                    model = LogisticRegression(solver='lbfgs')
                    model.fit(self.embd_tr, self.y_tr)
                    y_hat_val = model.predict(self.embd_val)
                    y_proba_val = model.predict_proba(self.embd_val)
                    acc = accuracy_score(self.y_val, y_hat_val)
                    prec = precision_score(self.y_val, y_hat_val)
                    recall = recall_score(self.y_val, y_hat_val)
                    auc = roc_auc_score(self.y_val, y_proba_val[:,1])
                    print('Accuracy {} precision {} recall {} AUC {}'.format(acc, prec, recall, auc))

                    if(acc>WARMUP_ACC):
                        if(use_base_group):
                            # combine initial training groups and new groups in next epoch training
                            new_tr_data, new_tr_size = self.hard_example_mining(y_hat_val, space, groupsTr, weightsTr)
                        else:
                            # abandom initial training groups and ONLY use new groups in next epoch training
                            new_tr_data, new_tr_size = self.hard_example_mining(y_hat_val, space)
                        self.run_epoch(sess, batchSize, new_tr_data, new_tr_size, True, epoch, train_writer)
                    else:
                        print('Abort hard example mining due to poor performance')
                        self.run_epoch(sess, batchSize, init_tr_data, init_tr_size, True, epoch, train_writer)

                saver.save(sess, os.path.join(currentModelPath+'/current', 'RECLE.ckpt'))

                # validation step
                if(epoch%2==0):
                    valLoss = self.run_epoch(sess, batchSize, val_data, val_size, False, epoch, val_writer)
                    if(valLoss<best_val_loss):
                        best_val_loss = valLoss
                        earlyStopCount = 0
                        saver.save(sess, os.path.join(currentModelPath, 'RECLE.ckpt'))
                        print('best loss model saved, epoch {}'.format(epoch))
                    if(valLoss>previous_val_loss):
                        earlyStopCount += 1
                    else:
                        earlyStopCount = 0
                    previous_val_loss = valLoss 
                    print('current early stop count {}, max count {}'.format(earlyStopCount, MAX_EARLY_STOP_COUNTS))
                    print('current best validation loss {}'.format(best_val_loss))

                    if(acc>best_acc):
                        best_acc = acc
                        saver.save(sess, os.path.join(currentModelPath+'/best_acc_model', 'RECLE.ckpt'))
                        print('best acc model saved, epoch {}'.format(epoch))


                    if((earlyStopCount>=MAX_EARLY_STOP_COUNTS or (valLoss-best_val_loss)>0.2) and epoch>50):
                        print('Early stop at epoch {}!'.format(epoch))
                        print('\n'*3)
                        break