import numpy as np
import os, time
import tensorflow as tf
import sys
import tensorflow.contrib.layers as layers
from utils import *

'''
This code implements the RLL framework.
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
class RLL(object):
    def __init__(self, dimension, l1_n, l2_n, gamma):
        self.dimension = dimension
        self.l1_n = l1_n
        self.l2_n = l2_n
        self.gamma = gamma
        self.current_train_step = 0
        self.current_val_step = 0
    
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
            print("Epoch {} train loss {}".format(epoch, total_loss))
        else:
            print('*'*60)
            print("Epoch {} validation loss {}".format(epoch, total_loss))
            print('*'*60)
            print('\n')
        return total_loss

    def train_and_evaluate(self, train, validation, batchSize, 
                            max_iter, save_path, summaries_dir, model_name, MAX_EARLY_STOP_COUNTS=5, val_groups_size=int(8e5)):   
        ratio = sum(train[:,0]) / train.shape[0]
        max_vote = max(train[:, 1])
        print('positive ratio {}, max vote {}'.format(ratio, max_vote))
        alpha = ratio*max_vote
        beta = max_vote - alpha
        
        # create validation groups and don't change it
        groupsVal, weightsVal = prepareInput(validation, groupSize=val_groups_size, alpha=alpha, beta=beta) 
        val_data = (groupsVal, weightsVal)
        val_size = groupsVal[0].shape[0]

        best_val_loss = sys.maxsize
        previous_val_loss = sys.maxsize
        earlyStopCount = 0
        saver = tf.train.Saver()
        print('training model {}'.format(model_name))

        currentModelPath = os.path.join(save_path, model_name)
        if(not os.path.exists(currentModelPath)):
            os.makedirs(currentModelPath)
        if(not os.path.exists(currentModelPath+'/current')):
            os.makedirs(currentModelPath+'/current')

        summaries_dir = os.path.join(summaries_dir, model_name)
        if(not os.path.exists(summaries_dir)):
            os.makedirs(summaries_dir)

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
            val_writer = tf.summary.FileWriter(summaries_dir + '/validation', sess.graph)
            tf.global_variables_initializer().run()

            for epoch in range(0, max_iter):
                # create different training groups for each epoch
                groupsTr, weightsTr = prepareInput(train, groupSize=int(1e5), alpha=alpha, beta=beta) 
                tr_data = (groupsTr, weightsTr)
                tr_size = groupsTr[0].shape[0]

                self.run_epoch(sess, batchSize, tr_data, tr_size, True, epoch, train_writer)
                saver.save(sess, os.path.join(currentModelPath+'/current', 'rll.ckpt'))

                if(epoch%2==0):
                    valLoss = self.run_epoch(sess, batchSize, val_data, val_size, False, epoch, val_writer)

                    if(valLoss<best_val_loss):
                        best_val_loss = valLoss
                        earlyStopCount = 0
                        saver.save(sess, os.path.join(currentModelPath, 'rll.ckpt'))
                        print('best model saved, epoch {}'.format(epoch))
                    if(valLoss>previous_val_loss):
                        earlyStopCount += 1
                    else:
                        earlyStopCount = 0
                    previous_val_loss = valLoss 
                    print('current early stop count {}, max count {}'.format(earlyStopCount, MAX_EARLY_STOP_COUNTS))
                    print('current best validation loss {}'.format(best_val_loss))
                
                    if((earlyStopCount>=MAX_EARLY_STOP_COUNTS or (valLoss-best_val_loss)>0.2) and epoch>50):
                        print('Early stop at epoch {}!'.format(epoch))
                        break