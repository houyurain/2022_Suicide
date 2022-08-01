"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 3_Transportability.py
@ Time: 8/1/22 10:45 AM
"""

import argparse
import time
from Utils import *
import random
from model import ml
from Combie_Features import CombineVocab


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--major_dataset', choices=['hidd', 'apcd', 'khin'], default='apcd')
    parser.add_argument('--transfer_dataset', choices=['hidd', 'apcd', 'khin'], default='khin')
    parser.add_argument('--run_model', choices=['LR', 'LIGHTGBM'], default='LR')
    parser.add_argument('--dump_detail', action='store_true', help='dump details of prediction')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/transfer/')
    args = parser.parse_args()

    args.output_dir = 'Transfer_performance/'
    args.major_dataset = 'khin'
    args.transfer_dataset = 'hidd'
    # args.run_model = 'LIGHTGBM'
    # args.dump_detail = True

    folder_name = args.major_dataset + '_' + args.transfer_dataset

    args.save_model_filename = os.path.join(args.output_dir, args.run_model, folder_name,
                                            'S{}_{}'.format(args.random_seed, args.run_model))
    check_and_mkdir(args.save_model_filename)

    return args


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    print('Major Dataset: ', args.major_dataset,
          'Transfer Dataset: ', args.transfer_dataset)

    encode2namefile = r'Datasets/icd9encode2name.pkl'

    majordatafile = r'Datasets/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(args.major_dataset, 'icd9')

    transdatafile = r'Datasets/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(args.transfer_dataset, 'icd9')
    print('encode2namefile:', encode2namefile)
    print('majordatafile:', majordatafile)
    print('transdatafile:', transdatafile)

    with open(encode2namefile, 'rb') as f:
        dx_name = pickle.load(f)
        print('len(dx_name):', len(dx_name))

    with open(majordatafile, 'rb') as f:
        major_data = pickle.load(f)
        print('len(major_data):', len(major_data))

    with open(transdatafile, 'rb') as f:
        transfer_data = pickle.load(f)
        print('len(transfer_data):', len(transfer_data))

    # Load combined features
    apcd_obj = open('Pre_train/selected_features_apcd.obj', 'rb')
    apcd_features = pickle.load(apcd_obj)
    apcd_obj.close()

    hidd_obj = open('Pre_train/selected_features_hidd.obj', 'rb')
    hidd_features = pickle.load(hidd_obj)
    hidd_obj.close()

    khin_obj = open('Pre_train/selected_features_khin.obj', 'rb')
    khin_features = pickle.load(khin_obj)
    khin_obj.close()

    combined_feature = CombineVocab(apcd_features, hidd_features, khin_features)

    my_dataset = Dataset(major_data, diag_name=dx_name, diag_code_vocab=combined_feature)
    x, y, uid_list, y_more = my_dataset.flatten_to_tensor(normalized_count=False, use_behavior=False)

    my_dataset_transfer = Dataset(transfer_data, diag_name=dx_name, diag_code_vocab=combined_feature)
    x_trans, y_trans, uid_list_trans, y_more_trans = my_dataset_transfer.flatten_to_tensor(normalized_count=False, use_behavior=False)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')

    train_ratio = 0.7  # 0.8  # 0.5
    val_ratio = 0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio,
          'test_ratio: ', 1 - (train_ratio + val_ratio))

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))

    dataset_transfer_size = len(my_dataset_transfer)
    indices_trans = list(range(dataset_transfer_size))

    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    train_trans_index = int(np.floor(train_ratio * dataset_transfer_size))
    val_trans_index = int(np.floor(val_ratio * dataset_transfer_size))
    np.random.shuffle(indices_trans)
    train_trans_indices, val_trans_indices, test_trans_indices = indices_trans[:train_trans_index], \
                                                                 indices_trans[
                                                                 train_trans_index:train_trans_index + val_trans_index], \
                                                                 indices_trans[train_trans_index + val_trans_index:]

    print("**************************************************")
    print(args.run_model, ' model learning:')


    def return_label(Y):
        # 1-visit-w/o-sui patient as negative
        Y[:, 0] = Y[:, 0] + Y[:, 2]
        return np.argmax(Y[:, :2], axis=1)


    print('Train data:')
    train_x, train_y, train_uid, train_y_more = x[train_indices], y[train_indices], uid_list[train_indices], y_more[
        train_indices]
    train_y = return_label(train_y)
    print('... pos ratio:', train_y.mean())
    print('Validation data:')
    val_x, val_y, val_uid, val_y_more = x[val_indices], y[val_indices], uid_list[val_indices], y_more[val_indices]
    val_y = return_label(val_y)
    print('... pos ratio:', val_y.mean())
    print('Test data:')
    test_x, test_y, test_uid, test_y_more = x[test_indices], y[test_indices], uid_list[test_indices], y_more[
        test_indices]
    test_y = return_label(test_y)
    print('... pos ratio:', test_y.mean())

    print('Train Transfer data:')
    train_x_trans, train_y_trans, train_uid_trans, train_y_more_trans = x_trans[train_trans_indices], y_trans[
        train_trans_indices], uid_list_trans[train_trans_indices], y_more_trans[train_trans_indices]
    train_y_trans = return_label(train_y_trans)
    print('... pos ratio:', train_y_trans.mean())
    print('Validation Transfer data:')
    val_x_trans, val_y_trans, val_uid_trans, val_y_more_trans = x_trans[val_trans_indices], y_trans[val_trans_indices], \
                                                                uid_list_trans[val_trans_indices], y_more_trans[
                                                                    val_trans_indices]
    val_y_trans = return_label(val_y_trans)
    print('... pos ratio:', val_y_trans.mean())
    print('Test Transfer data:')
    test_x_trans, test_y_trans, test_uid_trans, test_y_more_trans = x_trans[test_trans_indices], y_trans[
        test_trans_indices], uid_list_trans[test_trans_indices], y_more_trans[test_trans_indices]
    test_y_trans = return_label(test_y_trans)
    print('... pos ratio:', test_y_trans.mean())

    if args.run_model == 'LR':
        paras_grid = {
            'penalty': ['l1', 'l2'],
            'C': 10 ** np.arange(-2, 2, 0.2),
            'max_iter': [150],
            'random_state': [args.random_seed],
        }
    elif args.run_model == 'LIGHTGBM':
        paras_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 1],
            'num_leaves': np.arange(5, 50, 10),
            'min_child_samples': [50, 100, 150, 200, 250],
            'random_state': [args.random_seed],
        }

    print('args: ', args)
    model = ml.MLModels(args.run_model, paras_grid).fit(train_x, train_y, val_x, val_y)

    print('...Original data results: ')
    print('......Results at specificity 0.9:')
    result_1 = model.performance_at_specificity_or_threshold(test_x_trans, test_y_trans, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_2 = model.performance_at_specificity_or_threshold(test_x_trans, test_y_trans, specificity=0.95)

    df1 = pd.DataFrame([result_1 + (model.best_hyper_paras,), result_2 + (model.best_hyper_paras,)],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support", 'best_hyper_paras'],
                       index=['r_9', 'r_95'])
    df1.to_csv(os.path.join(os.path.dirname(args.save_model_filename),
                            'test_results_{}r{}.csv'.format(args.run_model, args.random_seed)))

    if args.run_model == 'LR':
        df3 = pd.DataFrame({'train_x_model': model.best_model.coef_[0]}, index=feature_name).reset_index()
        df3.to_csv(os.path.join(os.path.dirname(args.save_model_filename),
                                'model_coef_train_LRr{}.csv'.format(args.random_seed)))
    elif args.run_model == 'LIGHTGBM':
        df3 = pd.DataFrame(
            {'feature_importances_': model.best_model.feature_importances_,
             'feature': model.best_model.feature_name_,
             'feature_name': feature_name}).to_csv(os.path.join(os.path.dirname(args.save_model_filename),
                                                                'feature_importance_{}r{}.csv'.format(
                                                                    args.run_model, args.random_seed)))
    test_y_pre_prob_trans = model.predict_proba(test_x_trans)
    auc = roc_auc_score(test_y_trans, test_y_pre_prob_trans)
    threshold = result_2[1]
    test_y_pre_trans = (test_y_pre_prob_trans > threshold).astype(int)
    r = precision_recall_fscore_support(test_y_trans, test_y_pre_trans)
    print('precision_recall_fscore_support:\n', r)
    if args.dump_detail:
        feat = [';'.join(feature_name[np.nonzero(test_x_trans[i, :])[0]]) for i in range(len(test_x_trans))]
        pd.DataFrame(
            {'test_uid': test_uid_trans, 'test_y_pre_prob': test_y_pre_prob_trans, 'test_y_pre': test_y_pre_trans, 'test_y': test_y_trans,
             'feat': feat}).to_csv(os.path.join(os.path.dirname(args.save_model_filename),
                                                'test_pre_details_{}r{}.csv'.format(args.run_model, args.random_seed)))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))