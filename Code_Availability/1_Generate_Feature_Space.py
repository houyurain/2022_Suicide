"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 1_Generate_Feature_Space.py
@ Time: 8/1/22 10:13 AM
"""

import argparse
import time
from Utils import *
import random


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['apcd', 'hidd', 'khin'], default='hidd')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/')
    args = parser.parse_args()

    args.output_dir = 'Pre_train/'
    args.dataset = 'khin'

    check_and_mkdir(args.output_dir)

    return args


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    print('args: ', args)

    print('Encoding: icd 9 first 3 digits')
    encode2namefile = r'Datasets/icd9encode2name.pkl'
    majordatafile = r'Datasets/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(args.dataset, 'icd9')

    print('encode2namefile:', encode2namefile)
    print('majordatafile:', majordatafile)

    with open(encode2namefile, 'rb') as f:
        dx_name = pickle.load(f)
        print('len(dx_name):', len(dx_name))

    with open(majordatafile, 'rb') as f:
        data_1st_neg = pickle.load(f)
        print('len(data_1st_neg):', len(data_1st_neg))

    my_dataset = Dataset(data_1st_neg, diag_name=dx_name, diag_code_topk=300)
    # Dump features
    obj_path = os.path.join(os.path.dirname(args.output_dir), 'selected_features_{}.obj'.format(args.dataset))
    feature_obj = open(obj_path, 'wb')
    pickle.dump(my_dataset.diag_code_vocab, feature_obj)
    feature_obj.close()

    print('{} Done! Total Time used:'.format(args.dataset), time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))