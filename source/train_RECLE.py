from RECLE import RECLE
from utils import *
import numpy as np
import pandas as pd
import os

# first column is ground truth label, second column is crowdsourced votes (set to 1 if you have no crowdsourced labels)
train = pd.read_csv('../data/preschool/train.csv')
validation = pd.read_csv('../data/preschool/validation.csv')
train = np.array(train)
validation = np.array(validation)


# grid search for parameters
dimension = train.shape[1]-2
gamma = 10.0
max_iter = 500

save_path = '../preschool/RECLE/model'
summaries_dir = '../preschool/RECLE/summary'

if(not os.path.exists(save_path)):
    os.makedirs(save_path)
if(not os.path.exists(summaries_dir)):
    os.makedirs(summaries_dir)



l1_n_lst = [64, 128]
l2_n_lst = [32, 64]
dropout_rate_lst = [0.5, 0.3]
reg_scale_lst = [2.0, 5.0]
lr_rate_lst = [0.05]
batchSize_lst = [256, 512]
activation_lst = ['tanh', 'sigmoid']
distance_metric_lst = ['cosine', 'l1', 'l2']
space_lst = ['raw', 'embedding']
done_lst = [x for x in os.listdir(save_path)]


for dropout_rate in dropout_rate_lst:
    for l1_n in l1_n_lst:
        for l2_n in l2_n_lst:
            for reg_scale in reg_scale_lst:
                for batchSize in batchSize_lst:
                    for lr_rate in lr_rate_lst:
                        for activation in activation_lst:
                            for distance_metric in distance_metric_lst:
                                for space in space_lst:
                                    try:
                                        model_name = 'RECLE_l1_{}_l2_{}_lr_{}_penalty_{}_bs_{}_dropout_{}_activation_{}_metric_{}_space_{}'.format(\
                                            l1_n, l2_n, lr_rate, reg_scale, batchSize, dropout_rate, activation, distance_metric, space)

                                        if(l1_n<=l2_n or model_name in done_lst):
                                            print('pass model {}'.format(model_name))
                                            continue

                                        model = RECLE(dimension, l1_n, l2_n, gamma, distance_metric, train, validation)
                                        model.buildRLL(lr_rate, reg_scale, dropout_rate, activation)
                                        model.train_and_evaluate(train, validation, batchSize, 
                                                                max_iter, activation, save_path, summaries_dir, 
                                                                model_name, space=space, use_base_group=False)
                                    except Exception as e:
                                        print(e)