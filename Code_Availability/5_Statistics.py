"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: 5_Statistics.py
@ Time: 8/1/22 11:19 AM
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from Utils import *


E95_dict = {'E950': 'pos', 'E951': 'pos', 'E952': 'pos', 'E953': 'hang', 'E954': 'drown', 'E955': 'firearm',
            'E956': 'cut', 'E957': 'jump', 'E958': 'other', 'E959': 'other'}
V6284_dict = {}
for code in range(870, 888):
    V6284_dict[str(code)] = 'cut'
for code in range(890, 898):
    V6284_dict[str(code)] = 'cut'
for code in range(960, 990):
    V6284_dict[str(code)] = 'pos'
extra_dict = {'881': 'cut', '9947': 'pos'}
for code in range(960, 990):
    extra_dict[str(code)] = 'pos'
rule_3_2 = ['2908', '2909', '29383', '295', '2960', '29610', '29611', '29612', '29613', '29614'] + \
           [str(x) for x in range(29620, 29637)] + [str(x) for x in range(29630, 29637)] + \
           [str(x) for x in range(29640, 29647)] + [str(x) for x in range(29650, 29657)] + \
           [str(x) for x in range(29660, 29667)] + ['2967', '29680', '29681', '29682', '29689', '29690', '29699', '297'] + \
           [str(x) for x in range(2980, 2990)] + ['299', '3004', '301'] + [str(x) for x in range(3090, 3100)] + \
           ['311', '7801']


def check_suicide_type_list(dx_list):
    type_list = []
    for dx in dx_list:
        if dx[:4] in E95_dict.keys():
            sa_type = E95_dict[dx[:4]]
            if sa_type not in type_list:
                type_list.append(sa_type)

    for dx in dx_list:
        if (dx[:3] in V6284_dict.keys()) and ('V6284' in dx_list):
            sa_type = V6284_dict[dx[:3]]
            if sa_type not in type_list:
                type_list.append(sa_type)

    flag = False
    for dx in dx_list:
        for sa_code in rule_3_2:
            if sa_code == dx[:len(sa_code)]:
                flag = True
    if flag:
        for dx in dx_list:
            for key, value in extra_dict.items():
                if key == dx[:len(key)]:
                    sa_type = value
                    if sa_type not in type_list:
                        type_list.append(sa_type)
    return type_list


def process_age(age):
    if 10 <= age <= 14:
        return 'age10-14'
    elif 15 <= age <= 19:
        return 'age15-19'
    elif 20 <= age <= 24:
        return 'age20-24'
    else:
        return 'age20-24'


def process_sex(sex):
    if sex == 1:
        return 'Female'
    else:
        return 'Male'


def create_table1(dataset):
    with open('Data/final_pats_1st_neg_dict_' + dataset + '-icd9_new_labeled.pkl', 'rb') as f:
        uid_records = pickle.load(f)
    f.close()

    case_age = {'age10-14': 0, 'age15-19': 0, 'age20-24': 0}
    case_sex = {'Female': 0, 'Male': 0}
    case_t2e = []
    case_sui_type = {}
    control_age = {'age10-14': 0, 'age15-19': 0, 'age20-24': 0}
    control_sex = {'Female': 0, 'Male': 0}
    n_case = 0
    n_control = 0
    for uid, records in tqdm(uid_records.items()):
        dxs = []
        ddat = outcome = sex = age = None

        i = 0
        for rec in records:
            i += 1
            ddat, outcome, sex, age = rec[:4]
            dxs.append(list(set([x for x in rec[4:]])))
            if outcome:
                break

        if outcome:
            if len(dxs) > 1:
                n_case += 1
                age_sui = process_age(age)
                case_age[age_sui] = case_age[age_sui] + 1

                processed_sex = process_sex(sex)
                case_sex[processed_sex] = case_sex[processed_sex] + 1
                case_t2e.append((records[i - 1][0] - records[i - 2][0]).days)
                dx_list = dxs[-1]
                sa_type_list = check_suicide_type_list(dx_list)
                if len(sa_type_list) == 1:
                    sa_type = sa_type_list[0]
                else:
                    if set(sa_type_list) == set(['pos', 'other']):
                        sa_type = 'pos'
                    elif set(sa_type_list) == set(['cut', 'other']):
                        sa_type = 'cut'
                    elif set(sa_type_list) == set(['hang', 'other']):
                        sa_type = 'hang'
                    elif set(sa_type_list) == set(['jump', 'other']):
                        sa_type = 'jump'
                    elif set(sa_type_list) == set(['firearm', 'other']):
                        sa_type = 'firearm'
                    else:
                        sa_type = 'suicide type >= 2'
                if sa_type in case_sui_type:
                    temp_count = case_sui_type[sa_type]
                    case_sui_type[sa_type] = temp_count + 1
                else:
                    case_sui_type[sa_type] = 1
            else:
                continue
        else:
            n_control += 1
            processed_control_age = process_age(age)
            processed_control_sex = process_sex(sex)
            control_age[processed_control_age] = control_age[processed_control_age] + 1
            control_sex[processed_control_sex] = control_sex[processed_control_sex] + 1

    row_names = []
    records = []

    row_names.append('N')
    records.append([n_case, n_control])

    row_names.append('Sex - no. (%)')
    records.append([])
    sex_col = ['Female', 'Male']
    row_names.extend(sex_col)
    records.extend([['{:,} ({:.1f})'.format(case_sex[sex], (case_sex[sex]/n_case) * 100),
                    '{:,} ({:.1f})'.format(control_sex[sex], (control_sex[sex]/n_control) * 100)] for sex in sex_col])

    row_names.append('Age group (Suicide) - no. (%)')
    records.append([])
    age_col = ['age10-14', 'age15-19', 'age20-24']
    row_names.extend(age_col)
    records.extend([['{:,} ({:.1f})'.format(case_age[age], (case_age[age]/n_case) * 100),
                     '{:,} ({:.1f})'.format(control_age[age], (control_age[age] / n_control) * 100)] for age in age_col])

    row_names.append('Survival time (Days)')
    records.append(['{:.2f}'.format(np.mean(case_t2e)), '-'])

    row_names.append('Suicide attempt methods - no. (%)')
    records.append([])
    sui_type_col = case_sui_type.keys()
    row_names.extend(sui_type_col)
    records.extend(
        [['{:,} ({:.1f})'.format(case_sui_type[sui_type], (case_sui_type[sui_type] / n_case) * 100), '-'] for sui_type in sui_type_col])

    df = pd.DataFrame(records, columns=['Case', 'Control'], index=row_names)
    print(df)
    output_path = 'Statistics/'
    check_and_mkdir(output_path)
    df.to_csv(output_path + dataset + '_table1.csv')


def main():
    create_table1('apcd')


if __name__ == '__main__':
    main()