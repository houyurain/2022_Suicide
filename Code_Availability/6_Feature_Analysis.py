"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 6_Feature_Analysis.py
@ Time: 8/1/22 11:32 AM
"""

import pandas as pd
import numpy as np
from Utils import *
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
    df_1st = pd.read_csv('Datasets/' + dataset_1 + '_matrix.csv')
    df_2nd = pd.read_csv('Datasets/' + dataset_2 + '_matrix.csv')
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
    output_path = 'Statistics/'
    check_and_mkdir(output_path)
    res.to_csv(output_path + dataset_1 + '_' + dataset_2 + '_smd_res.csv', index=False)


def LR_coef_smd_plot(dataset_1, dataset_2):
    smd_data = pd.read_csv('Statistics/' + dataset_1 + '_' + dataset_2 + '_smd_res.csv')
    smd_data_dict = smd_data.set_index(['feature'])[dataset_1 + '_' + dataset_2 + '_smd'].to_dict()
    smd_dict = {}
    for key, value in smd_data_dict.items():
        if '$' in key:
            smd_dict[key.split('$')[1]] = value
    smd_dict['Age10-14'] = smd_data_dict['age10-14']
    smd_dict['Age15-19'] = smd_data_dict['age15-19']
    smd_dict['Age20-24'] = smd_data_dict['age20-24']
    smd_dict['Female'] = smd_data_dict['sex']
    smd_data = smd_data[['feature', dataset_1 + '_' + dataset_2 + '_smd']]

    coef_1_data = pd.read_csv('Results/Local_performance/' + dataset_1 + '_coef_res.csv')
    coef_2_data = pd.read_csv('Results/Local_performance/' + dataset_2 + '_coef_res.csv')

    coef_1_df = coef_1_data.rename(columns={'index': 'feature', 'coef_mean': 'coef_1', 'coef_ci_std': 'coef_ci_1'})
    coef_2_df = coef_2_data.rename(columns={'index': 'feature', 'coef_mean': 'coef_2', 'coef_ci_std': 'coef_ci_2'})
    coef_1_df = coef_1_df[['feature', 'coef_1', 'coef_ci_1']]
    coef_2_df = coef_2_df[['feature', 'coef_2', 'coef_ci_2']]

    coef_1_df = coef_1_df.sort_values(by=['coef_1'])
    coef_1_df = coef_1_df.reset_index(drop=True)

    fig_data = pd.merge(coef_1_df, coef_2_df, on='feature')
    fig_data = pd.merge(fig_data, smd_data, on='feature')
    fig_data = fig_data.reset_index(drop=True)
    for i in range(len(fig_data)):
        fea_name = fig_data.loc[i, 'feature']
        fea_name = 'female' if fea_name == 'sex' else fea_name
        fig_data.loc[i, 'feature'] = fea_name.split('$')[1] if '$' in fea_name else fea_name.capitalize()

    fig, axes = plt.subplots(figsize=(21, 25))

    color_dict = {'apcd': '#e07a5f', 'hidd': '#81b29a', 'khin': '#ffcb77'}
    width = 0.25
    ind = np.arange(len(fig_data))
    axes.barh(ind, fig_data['coef_2'], width, xerr=fig_data['coef_ci_2'], color=color_dict[dataset_2], alpha=.8)
    axes.barh(ind + width, fig_data['coef_1'], width, xerr=fig_data['coef_ci_1'], color=color_dict[dataset_1], alpha=.8)
    axes.scatter(fig_data[dataset_1 + '_' + dataset_2 + '_smd'], ind + width, s=45, color='#9b5de5')

    axes.axvline(x=0, color='black')
    axes.axvline(x=0.2, linestyle='dashed', color='gray')
    axes.axvline(x=-0.2, linestyle='dashed', color='gray')

    axes.text(0.2, -1, "0.2", ha="center", va="top", color='red', fontsize=16)
    axes.text(-0.2, -1, "-0.2", ha="center", va="top", color='red', fontsize=16)

    y_labels = ['\n'.join(wrap(fea, 70)) for fea in fig_data['feature']]
    axes.set(yticks=ind + width, yticklabels=y_labels, ylim=[2 * width - 1, len(smd_data)])
    # [t.set_color('red') for t in axes.yaxis.get_ticklabels() if (smd_dict[t.get_text().replace('\n', ' ')] > 0.2) or (smd_dict[t.get_text().replace('\n', ' ')] < -0.2)]
    axes.yaxis.grid()

    label_dict = {'apcd': 'APCD', 'hidd': 'HIDD', 'khin': 'KHIN'}
    handle_list = [mpatches.Patch(color=color_dict[dataset_1], label=label_dict[dataset_1]),
                   mpatches.Patch(color=color_dict[dataset_2], label=label_dict[dataset_2]),
                   mpatches.Patch(color='#9b5de5',
                                  label='SMD({}, {})'.format(label_dict[dataset_1], label_dict[dataset_2]))]

    if dataset_1 == 'apcd':
        plt.xlim([-3, 2])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(handles=handle_list, prop={'size': 12}, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.subplots_adjust(left=.41, right=.87)
    # plt.show()
    output_path = 'Investigation/Figures/'
    check_and_mkdir(output_path)
    plt.savefig(output_path + dataset_1 + '_' + dataset_2 + '_LR.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path + dataset_1 + '_' + dataset_2 + '_LR.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    dataset_1 = 'apcd'
    dataset_2 = 'hidd'
    # smd_in_two(dataset_1, dataset_2)
    LR_coef_smd_plot(dataset_1, dataset_2)


if __name__ == '__main__':
    main()