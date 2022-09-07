"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 3_2_Transfer_LSTM.py
@ Time: 9/6/22 4:51 PM
"""

import copy
import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse

from utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn.functional as F
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from model import lstm, ml
import itertools
import functools
from Dataset_LSTM import LSTM_Dataset
from Vocab import *


def transferred_performance_from_pretrained_lstm(dataset_name, random_seed):
    start_time = time.time()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    if dataset_name == 'apcd':
        target_1 = 'hidd'
        target_2 = 'khin'
    elif dataset_name == 'hidd':
        target_1 = 'apcd'
        target_2 = 'khin'
    elif dataset_name == 'khin':
        target_1 = 'apcd'
        target_2 = 'hidd'

    output_dir = 'Transfer_performance/LSTM/'
    save_model_filename_1 = os.path.join(output_dir, '{}/'.format(dataset_name + '_' + target_1))
    save_model_filename_2 = os.path.join(output_dir, '{}/'.format(dataset_name + '_' + target_2))
    check_and_mkdir(save_model_filename_1)
    check_and_mkdir(save_model_filename_2)

    encode2namefile = r'dataset/icd9encode2name.pkl'
    target_1datafile = r'dataset/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(target_1, 'icd9')
    target_2datafile = r'dataset/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(target_2, 'icd9')

    with open(encode2namefile, 'rb') as f:
        dx_name = pickle.load(f)

    with open(target_1datafile, 'rb') as f:
        target_1_data = pickle.load(f)

    with open(target_2datafile, 'rb') as f:
        target_2_data = pickle.load(f)

    with open('Pre_train/selected_features_apcd.obj', 'rb') as f:
        vocab_apcd = pickle.load(f)

    with open('Pre_train/selected_features_hidd.obj', 'rb') as f:
        vocab_hidd = pickle.load(f)

    with open('Pre_train/selected_features_khin.obj', 'rb') as f:
        vocab_khin = pickle.load(f)

    vocab_combined = copy.deepcopy(vocab_apcd)
    vocab_combined.extend_vocab(vocab_hidd)
    vocab_combined.extend_vocab(vocab_khin)

    print('Using combined feature space')
    my_target_1 = LSTM_Dataset(target_1_data, diag_name=dx_name, diag_code_vocab=vocab_combined)
    my_target_2 = LSTM_Dataset(target_2_data, diag_name=dx_name, diag_code_vocab=vocab_combined)

    train_ratio = 0.8
    print('train_ratio: ', train_ratio,
          'test_ratio: ', 1 - train_ratio)

    # Transferred Dataset 1
    dataset_target_1_size = len(my_target_1)
    indices_target_1 = list(range(dataset_target_1_size))
    train_target_1_index = int(np.floor(train_ratio * dataset_target_1_size))
    np.random.shuffle(indices_target_1)
    test_target_1_indices = indices_target_1[train_target_1_index:]

    test_target_1_sampler = SubsetRandomSampler(test_target_1_indices)
    test_target_1_loader = torch.utils.data.DataLoader(my_target_1, batch_size=256, sampler=test_target_1_sampler)

    # Transferred Dataset 2
    dataset_target_2_size = len(my_target_2)
    indices_target_2 = list(range(dataset_target_2_size))
    train_target_2_index = int(np.floor(train_ratio * dataset_target_2_size))
    np.random.shuffle(indices_target_2)
    test_target_2_indices = indices_target_2[train_target_2_index:]

    test_target_2_sampler = SubsetRandomSampler(test_target_2_indices)
    test_target_2_loader = torch.utils.data.DataLoader(my_target_2, batch_size=256, sampler=test_target_2_sampler)

    pretrained_model = load_model(lstm.LSTMModel, 'output/5fold/lstm/' + dataset_name + '/lstm_S' + str(random_seed) + '.model')

    auc_test_1, loss_test_1, Y_test_1, Y_pred_test_1, uid_test_1 = transfer_data(pretrained_model, test_target_1_loader, cuda=True, normalized=False)
    auc_test_2, loss_test_2, Y_test_2, Y_pred_test_2, uid_test_2 = transfer_data(pretrained_model, test_target_2_loader, cuda=True, normalized=False)

    print('Final results', 'loss_test', loss_test_1, "test-auc_1", auc_test_1)
    print('Final results', 'loss_test', loss_test_2, "test-auc_2", auc_test_2)

    result_1_1 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test_1, Y_test_1, specificity=0.9)
    result_1_2 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test_1, Y_test_1, specificity=0.95)

    result_2_1 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test_2, Y_test_2, specificity=0.9)
    result_2_2 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test_2, Y_test_2, specificity=0.95)

    df1 = pd.DataFrame([result_1_1, result_1_2],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support"],
                       index=['r_9', 'r_95'])
    df1.to_csv(os.path.join(os.path.dirname(save_model_filename_1), 'test_results_{}r{}.csv'.format('LSTM', random_seed)))

    df2 = pd.DataFrame([result_2_1, result_2_2],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support"],
                       index=['r_9', 'r_95'])
    df2.to_csv(os.path.join(os.path.dirname(save_model_filename_2), 'test_results_{}r{}.csv'.format('LSTM', random_seed)))


    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def main():
    for dataset_name in ['apcd', 'hidd', 'khin']:
        for random_seed in range(10):
            transferred_performance_from_pretrained_lstm(dataset_name, random_seed)


if __name__ == '__main__':
    main()