"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 3_1_Transfer.py
@ Time: 9/7/22 9:57 AM
"""

import time
from Utils import *
from Combie_Features import CombineVocab


def transferred_performance_from_pretrained(dataset_name, run_model, random_seed):
    start_time = time.time()
    if dataset_name == 'apcd':
        target_1 = 'hidd'
        target_2 = 'khin'
    elif dataset_name == 'hidd':
        target_1 = 'apcd'
        target_2 = 'khin'
    elif dataset_name == 'khin':
        target_1 = 'apcd'
        target_2 = 'hidd'

    target_dataset_1 = pd.read_csv('Datasets/' + target_1 + '_matrix.csv')
    target_dataset_2 = pd.read_csv('Datasets/' + target_2 + '_matrix.csv')
    features = target_dataset_1.columns.tolist()[1:-1]

    X_target_1, y_target_1 = target_dataset_1[features].values, target_dataset_1['label'].values
    X_target_2, y_target_2 = target_dataset_2[features].values, target_dataset_2['label'].values

    train_ratio = 0.8
    print('train_ratio: ', train_ratio,
          'test_ratio: ', 1 - train_ratio)

    # Transferred Dataset 1
    dataset_target_1_size = len(target_dataset_1)
    indices_target_1 = list(range(dataset_target_1_size))
    train_target_1_index = int(np.floor(train_ratio * dataset_target_1_size))
    np.random.shuffle(indices_target_1)
    test_target_1_indices = indices_target_1[train_target_1_index:]
    test_x_target_1, test_y_target_1 = X_target_1[test_target_1_indices], y_target_1[test_target_1_indices]

    # Transferred Dataset 2
    dataset_target_2_size = len(target_dataset_2)
    indices_target_2 = list(range(dataset_target_2_size))
    train_target_2_index = int(np.floor(train_ratio * dataset_target_2_size))
    np.random.shuffle(indices_target_2)
    test_target_2_indices = indices_target_2[train_target_2_index:]
    test_x_target_2, test_y_target_2 = X_target_2[test_target_2_indices], y_target_2[test_target_2_indices]

    model_path = 'Local_performance/' + run_model + '/' + dataset_name + '/'
    with open(model_path + dataset_name + '_' + run_model + '_' + str(random_seed) + '.model', 'rb') as f:
        pretrained_model = pickle.load(f)
    f.close()

    result_1_1 = pretrained_model.performance_at_specificity_or_threshold(test_x_target_1, test_y_target_1, specificity=0.9)
    result_1_2 = pretrained_model.performance_at_specificity_or_threshold(test_x_target_1, test_y_target_1, specificity=0.95)

    result_2_1 = pretrained_model.performance_at_specificity_or_threshold(test_x_target_2, test_y_target_2, specificity=0.9)
    result_2_2 = pretrained_model.performance_at_specificity_or_threshold(test_x_target_2, test_y_target_2, specificity=0.95)

    output_dir = 'Transfer_performance/' + run_model + '/'
    save_model_filename_1 = os.path.join(output_dir, '{}/'.format(dataset_name + '_' + target_1))
    save_model_filename_2 = os.path.join(output_dir, '{}/'.format(dataset_name + '_' + target_2))
    check_and_mkdir(save_model_filename_1)
    check_and_mkdir(save_model_filename_2)

    df1 = pd.DataFrame([result_1_1, result_1_2],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support"],
                       index=['r_9', 'r_95'])
    df1.to_csv(os.path.join(os.path.dirname(save_model_filename_1), 'test_results_{}r{}.csv'.format(run_model, random_seed)))

    df2 = pd.DataFrame([result_2_1, result_2_2],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support"],
                       index=['r_9', 'r_95'])
    df2.to_csv(os.path.join(os.path.dirname(save_model_filename_2), 'test_results_{}r{}.csv'.format(run_model, random_seed)))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def main():
    run_model = 'LIGHTGBM'
    for dataset_name in ['apcd', 'hidd', 'khin']:
        for random_seed in range(20):
            transferred_performance_from_pretrained(dataset_name, run_model, random_seed)


if __name__ == '__main__':
    main()