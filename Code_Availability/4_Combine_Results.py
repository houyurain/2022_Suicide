"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 4_Combine_Results.py
@ Time: 8/1/22 10:54 AM
"""

import pandas as pd
import numpy as np
from Utils import *


def boot_matrix(z, B):
    """Bootstrap sample
    Returns all bootstrap samples in a matrix"""
    z = np.array(z).flatten()
    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean_ci(x, B=1000, alpha=0.05):
    n = len(x)
    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)
    quantile_confidence_interval = np.percentile(sampling_distribution, q=(100 * alpha / 2, 100 * (1 - alpha / 2)))
    std = sampling_distribution.std()
    # if plot:
    #     plt.hist(sampling_distribution, bins="fd")
    return quantile_confidence_interval, std


def combine_local_performance_results(learner, dataset_name):
    auc_list = []
    sen_9_list = []
    ppv_9_list = []
    sen_95_list = []
    ppv_95_list = []
    df_path = 'Local_performance/' + learner + '/' + dataset_name + '/'
    output_path = 'Results/Local_performance/'
    check_and_mkdir(output_path)

    for rnd_seed in range(50):
        res_df = pd.read_csv(df_path + 'test_results_' + learner + 'r' + str(rnd_seed) + '.csv')
        res_df = res_df.set_index(['Unnamed: 0'])
        auc = res_df.loc['r_9', 'AUC']
        auc_list.append(auc)
        sen_9 = res_df.loc['r_9', 'Sensitivity/recall']
        ppv_9 = res_df.loc['r_9', 'PPV/precision']
        sen_9_list.append(sen_9)
        ppv_9_list.append(ppv_9)
        sen_95 = res_df.loc['r_95', 'Sensitivity/recall']
        ppv_95 = res_df.loc['r_95', 'PPV/precision']
        sen_95_list.append(sen_95)
        ppv_95_list.append(ppv_95)

    auc_ci_res = bootstrap_mean_ci(auc_list)
    auc_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(auc_list), auc_ci_res[0][0], auc_ci_res[0][1])
    sen_9_ci_res = bootstrap_mean_ci(sen_9_list)
    sen_9_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(sen_9_list), sen_9_ci_res[0][0], sen_9_ci_res[0][1])
    sen_95_ci_res = bootstrap_mean_ci(sen_95_list)
    sen_95_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(sen_95_list), sen_95_ci_res[0][0], sen_95_ci_res[0][1])
    ppv_9_ci_res = bootstrap_mean_ci(ppv_9_list)
    ppv_9_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(ppv_9_list), ppv_9_ci_res[0][0], ppv_9_ci_res[0][1])
    ppv_95_ci_res = bootstrap_mean_ci(ppv_95_list)
    ppv_95_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(ppv_95_list), ppv_95_ci_res[0][0], ppv_95_ci_res[0][1])
    res = pd.DataFrame(columns=['Dataset', 'Transferred Dataset', 'AUC (95CI)', 'Sensitivity (0.9)', 'PPV (0.9)', 'Sensitivity (0.95)', 'PPV (0.95)'])
    res.loc[0] = [dataset_name, '-', auc_res, sen_9_res, ppv_9_res, sen_95_res, ppv_95_res]
    # print(res[['AUC (95CI)']])
    res.to_csv(output_path + dataset_name + '_' + learner + '_local_performance_res.csv', index=False)


def combine_coef_results(dataset_name):
    data_path = 'Local_performance/LR/' + dataset_name + '/'
    output_path = 'Results/Local_performance/'
    check_and_mkdir(output_path)

    coef_data = pd.read_csv(data_path + 'model_coef_train_LRr0.csv')
    coef_data = coef_data.rename(columns={'train_x_model': 'RS_0'})
    coef_data = coef_data[['index', 'RS_0']]
    rs_list = ['RS_0']
    for rs in range(1, 50):
        temp_coef_data = pd.read_csv(data_path + 'model_coef_train_LRr' + str(rs) + '.csv')
        temp_coef_data = temp_coef_data.rename(columns={'train_x_model': 'RS_' + str(rs)})
        temp_coef_data = temp_coef_data[['index', 'RS_' + str(rs)]]
        coef_data = pd.merge(coef_data, temp_coef_data, on='index')
        rs_list.append('RS_' + str(rs))
    coef_mean = coef_data[rs_list].mean(axis=1).tolist()
    coef_std = coef_data[rs_list].std(axis=1).tolist()
    coef_data['coef_mean'] = coef_mean
    coef_data['coef_std'] = coef_std

    coef_rs_df = coef_data[rs_list]
    coef_rs_df = coef_rs_df.reset_index(drop=True)
    coef_ci_list = []
    coef_ci_std = []
    for i in range(len(coef_rs_df)):
        fea_coefs = coef_rs_df.iloc[i].values
        ci_res = bootstrap_mean_ci(fea_coefs)
        coef_ci_list.append('({:.3f}, {:.3f})'.format(ci_res[0][0], ci_res[0][1]))
        coef_ci_std.append('{:.4f}'.format(ci_res[1]))
    coef_data['coef_ci'] = coef_ci_list
    coef_data['coef_ci_std'] = coef_ci_std
    coef_data = coef_data[['index', 'coef_mean', 'coef_std', 'coef_ci', 'coef_ci_std']]
    # print(coef_data)
    coef_data.to_csv(output_path + dataset_name + '_coef_res.csv', index=False)


def combine_gbm_featureImportans_results(dataset_name):
    data_path = 'Local_performance/LIGHTGBM/' + dataset_name + '/'
    output_path = 'Results/Local_performance/'
    check_and_mkdir(output_path)

    featureImportans_data = pd.read_csv(data_path + 'feature_importance_LIGHTGBMr0.csv')
    featureImportans_data['feature_importances_norm'] = (featureImportans_data['feature_importances_'] - featureImportans_data['feature_importances_'].min()) / (featureImportans_data['feature_importances_'].max() - featureImportans_data['feature_importances_'].min())
    featureImportans_data = featureImportans_data.rename(columns={'feature_importances_norm': 'RS_0'})
    featureImportans_data = featureImportans_data[['feature_name', 'RS_0']]
    rs_list = ['RS_0']
    for rs in range(1, 50):
        temp_featureImportans_data = pd.read_csv(data_path + 'feature_importance_LIGHTGBMr' + str(rs) + '.csv')
        temp_featureImportans_data['feature_importances_norm'] = (temp_featureImportans_data['feature_importances_'] - temp_featureImportans_data['feature_importances_'].min()) / (temp_featureImportans_data['feature_importances_'].max() - temp_featureImportans_data['feature_importances_'].min())
        temp_featureImportans_data = temp_featureImportans_data.rename(columns={'feature_importances_norm': 'RS_' + str(rs)})
        temp_featureImportans_data = temp_featureImportans_data[['feature_name', 'RS_' + str(rs)]]
        featureImportans_data = pd.merge(featureImportans_data, temp_featureImportans_data, on='feature_name')
        rs_list.append('RS_' + str(rs))
    featureImportans_mean = featureImportans_data[rs_list].mean(axis=1).tolist()
    featureImportans_std = featureImportans_data[rs_list].std(axis=1).tolist()
    featureImportans_data['feature_importance_mean'] = featureImportans_mean
    featureImportans_data['feature_importance_std'] = featureImportans_std

    featureImportans_df = featureImportans_data[rs_list]
    featureImportans_df = featureImportans_df.reset_index(drop=True)
    featureImportans_ci_list = []
    featureImportans_ci_std = []
    for i in range(len(featureImportans_df)):
        fea_featureImportans = featureImportans_df.iloc[i].values
        ci_res = bootstrap_mean_ci(fea_featureImportans)
        featureImportans_ci_list.append('({:.3f}, {:.3f})'.format(ci_res[0][0], ci_res[0][1]))
        featureImportans_ci_std.append('{:.4f}'.format(ci_res[1]))
    featureImportans_data['feature_importance_ci'] = featureImportans_ci_list
    featureImportans_data['feature_importance_ci_std'] = featureImportans_ci_std
    featureImportans_data = featureImportans_data[['feature_name', 'feature_importance_mean', 'feature_importance_std',
                                                   'feature_importance_ci', 'feature_importance_ci_std']]

    featureImportans_data.to_csv(output_path + dataset_name + '_featureImportans_res.csv', index=False)


def combine_transfer_performance_results(learner, dataset_name):
    dataset_name_1, dataset_name_2 = dataset_name.split('_')
    auc_list = []
    sen_9_list = []
    ppv_9_list = []
    sen_95_list = []
    ppv_95_list = []
    df_path = 'Transfer_performance/' + learner + '/' + dataset_name + '/'
    output_path = 'Results/Transfer_performance/'
    check_and_mkdir(output_path)

    for rnd_seed in range(50):
        res_df = pd.read_csv(df_path + 'test_results_' + learner + 'r' + str(rnd_seed) + '.csv')
        res_df = res_df.set_index(['Unnamed: 0'])
        auc = res_df.loc['r_9', 'AUC']
        auc_list.append(auc)
        sen_9 = res_df.loc['r_9', 'Sensitivity/recall']
        ppv_9 = res_df.loc['r_9', 'PPV/precision']
        sen_9_list.append(sen_9)
        ppv_9_list.append(ppv_9)
        sen_95 = res_df.loc['r_95', 'Sensitivity/recall']
        ppv_95 = res_df.loc['r_95', 'PPV/precision']
        sen_95_list.append(sen_95)
        ppv_95_list.append(ppv_95)

    auc_ci_res = bootstrap_mean_ci(auc_list)
    auc_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(auc_list), auc_ci_res[0][0], auc_ci_res[0][1])
    sen_9_ci_res = bootstrap_mean_ci(sen_9_list)
    sen_9_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(sen_9_list), sen_9_ci_res[0][0], sen_9_ci_res[0][1])
    sen_95_ci_res = bootstrap_mean_ci(sen_95_list)
    sen_95_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(sen_95_list), sen_95_ci_res[0][0], sen_95_ci_res[0][1])
    ppv_9_ci_res = bootstrap_mean_ci(ppv_9_list)
    ppv_9_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(ppv_9_list), ppv_9_ci_res[0][0], ppv_9_ci_res[0][1])
    ppv_95_ci_res = bootstrap_mean_ci(ppv_95_list)
    ppv_95_res = '{:.3f} ({:.3f}, {:.3f})'.format(np.mean(ppv_95_list), ppv_95_ci_res[0][0], ppv_95_ci_res[0][1])
    res = pd.DataFrame(columns=['Dataset', 'Transferred Dataset', 'AUC (95CI)', 'Sensitivity (0.9)', 'PPV (0.9)', 'Sensitivity (0.95)', 'PPV (0.95)'])
    res.loc[0] = [dataset_name_1, dataset_name_2, auc_res, sen_9_res, ppv_9_res, sen_95_res, ppv_95_res]
    # print(res[['AUC (95CI)']])
    res.to_csv(output_path + dataset_name + '_' + learner + '_transfer_performance_res.csv', index=False)


def combine_final_res(learner):
    local_path = 'Results/Local_performance/'
    trans_path = 'Results/Transfer_performance/'
    dataset_list = ['apcd', 'hidd', 'khin']
    res = pd.DataFrame()
    for dataset in dataset_list:
        res_path = local_path + dataset + '_' + learner + '_performance_res.csv'
        res_data = pd.read_csv(res_path)
        res = pd.concat((res, res_data))

    for dataset_1 in dataset_list:
        for dataset_2 in dataset_list:
            if dataset_1 != dataset_2:
                dataset = dataset_1 + '_' + dataset_2
                res_path = trans_path + dataset + '_' + learner + '_performance_res.csv'
                res_data = pd.read_csv(res_path)
                res = pd.concat((res, res_data))
    print(res)
    res.to_csv('Results/' + learner + '_total_res.csv', index=False)


def main():
    # combine_local_performance_results('LR', 'apcd')
    # combine_coef_results('apcd')
    # combine_gbm_featureImportans_results('apcd')

    # combine_transfer_performance_results('LR', 'apcd_hidd')

    combine_final_res('LR')


if __name__ == '__main__':
    main()