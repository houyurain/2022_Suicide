"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 6_Feature_Analysis.py
@ Time: 8/1/22 11:32 AM
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textwrap import wrap


def _percentage_str(x):
    n = x.sum()
    per = x.mean()
    return '{:,} ({:.1f})'.format(n, per * 100)


def _smd(x1, x2):
    m1 = x1.mean()
    m2 = x2.mean()
    v1 = x1.var()
    v2 = x2.var()

    VAR = np.sqrt((v1 + v2) / 2)
    smd = np.divide(m1 - m2, VAR, out=np.zeros_like(m1), where=VAR != 0)

    return '{:.3f}'.format(smd.astype(float))


def top_features_in_3dataset_coef():
    dataset_list = ['apcd', 'hidd', 'khin']

    pos_top_fea = []
    neg_top_fea = []
    for dataset_name in dataset_list:
        coef_data = pd.read_csv(folder + 'model/' + dataset_name + '_coef_res.csv')
        coef_data = coef_data.sort_values(by=['coef_mean'], ascending=False)
        pos_top10 = coef_data['index'].tolist()[:10]
        neg_top10 = coef_data['index'].tolist()[-10:]
        for i in range(10):
            pos_fea = pos_top10[i]
            neg_fea = neg_top10[i]
            if (pos_fea not in pos_top_fea) and ('sex' not in pos_fea) and ('age' not in pos_fea):
                pos_top_fea.append(pos_fea)
            if (neg_fea not in neg_top_fea) and ('sex' not in neg_fea) and ('age' not in neg_fea):
                neg_top_fea.append(neg_fea)

    return pos_top_fea, neg_top_fea


def smd_in_two_dataset_coef(dataset_1, dataset_2):
    df_1st = pd.read_csv(folder + 'dataset/' + dataset_1 + '_matrix.csv')
    df_2nd = pd.read_csv(folder + 'dataset/' + dataset_2 + '_matrix.csv')
    pos_top_fea, neg_top_fea = top_features_in_3dataset_coef()
    demo_list = ['sex', 'age10-14', 'age15-19', 'age20-24']
    res = pd.DataFrame(columns=['feature', dataset_1, dataset_2, dataset_1 + '_' + dataset_2 + '_smd',
                                dataset_2 + '_' + dataset_1 + '_smd'])
    idx = 0

    for fea in demo_list:
        smd_1 = _smd(df_1st[fea], df_2nd[fea])
        smd_2 = _smd(df_2nd[fea], df_1st[fea])
        res.loc[idx] = [fea, _percentage_str(df_1st[fea]), _percentage_str(df_2nd[fea]), smd_1, smd_2]
        idx += 1

    for fea in pos_top_fea:
        smd_1 = _smd(df_1st[fea], df_2nd[fea])
        smd_2 = _smd(df_2nd[fea], df_1st[fea])
        res.loc[idx] = [fea, _percentage_str(df_1st[fea]), _percentage_str(df_2nd[fea]), smd_1, smd_2]
        idx += 1

    for fea in neg_top_fea:
        smd_1 = _smd(df_1st[fea], df_2nd[fea])
        smd_2 = _smd(df_2nd[fea], df_1st[fea])
        res.loc[idx] = [fea, _percentage_str(df_1st[fea]), _percentage_str(df_2nd[fea]), smd_1, smd_2]
        idx += 1

    print(res)
    res.to_csv(folder + 'statistics/' + dataset_1 + '_' + dataset_2 + '_smd_res.csv', index=False)