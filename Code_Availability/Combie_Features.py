"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@ File: Combie_Features.py
@ Time: 4/20/22 9:16 PM
"""

import pickle


class CombineVocab(object):
    END_CODE = '<end>'
    PAD_CODE = '<pad>'
    UNK_CODE = '<unk>'

    def __init__(self, dataset_1, dataset_2, dataset_3):
        super().__init__()

        self.code2id_1 = dataset_1.code2id
        self.id2code_1 = dataset_1.id2code
        self.code2name_1 = dataset_1.code2name
        self.id2name_1 = dataset_1.id2name

        self.threshold_1 = dataset_1.threshold
        self.topk_1 = dataset_1.topk
        self.code2count_1 = dataset_1.code2count

        self.code2id_2 = dataset_2.code2id
        self.id2code_2 = dataset_2.id2code
        self.code2name_2 = dataset_2.code2name
        self.id2name_2 = dataset_2.id2name

        self.threshold_2 = dataset_2.threshold
        self.topk_2 = dataset_2.topk
        self.code2count_2 = dataset_2.code2count

        self.code2id_3 = dataset_3.code2id
        self.id2code_3 = dataset_3.id2code
        self.code2name_3 = dataset_3.code2name
        self.id2name_3 = dataset_3.id2name

        self.threshold_3 = dataset_3.threshold
        self.topk_3 = dataset_3.topk
        self.code2count_3 = dataset_3.code2count

        self.id2code = self.id2code_1
        self.id2name = self.id2name_1
        self.code2id = self.code2id_1
        self.code_source = ['apcd'] * len(self.id2code)

        index = len(self.id2code)
        for id_2 in self.id2code_2:
            id2code_2_value = self.id2code_2[id_2]
            id2name_2_name = self.id2name_2[id_2]
            if id2code_2_value not in self.id2code.values():
                self.id2code[index] = id2code_2_value
                self.id2name[index] = id2name_2_name
                self.code2id[id2code_2_value] = index
                self.code_source.append('hidd')
                index += 1

        for id_3 in self.id2code_3:
            id2code_3_value = self.id2code_3[id_3]
            id2name_3_name = self.id2name_3[id_3]
            if id2code_3_value not in self.id2code.values():
                self.id2code[index] = id2code_3_value
                self.id2name[index] = id2name_3_name
                self.code2id[id2code_3_value] = index
                self.code_source.append('khin')
                index += 1

    def get(self, item, default=None):
        return self.code2id.get(item, default)

    def feature_name(self):
        return [self.id2name.get(x, '') for x in range(len(self.code2id))]

    def __getitem__(self, item):
        return self.code2id[item]

    def __contains__(self, item):
        return item in self.code2id

    def __len__(self):
        return len(self.code2id)

    def __str__(self):
        return f'{len(self)} codes'


def combine_APCD_HIDD():
    apcd_obj = open('/Pre_train/selected_features_apcd.obj', 'rb')
    apcd_features = pickle.load(apcd_obj)
    apcd_obj.close()

    hidd_obj = open('/Pre_train/selected_features_hidd.obj', 'rb')
    hidd_features = pickle.load(hidd_obj)
    hidd_obj.close()

    khin_obj = open('/Pre_train/selected_features_khin.obj', 'rb')
    khin_features = pickle.load(khin_obj)
    khin_obj.close()

    combine_fea = CombineVocab(apcd_features, hidd_features, khin_features)
    feature_obj = open('/Pre_train/combined_features.obj', 'wb')
    pickle.dump(combine_fea, feature_obj)
    feature_obj.close()


def main():
    combine_APCD_HIDD()


if __name__ == '__main__':
    main()
