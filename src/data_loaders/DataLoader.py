# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
from utils.dataset import group_user_interactions_df
from utils.global_p import *
import json


class DataLoader(object):
    """
    只负责load数据集文件，记录一些数据集信息
    """

    @staticmethod
    def parse_data_args(parser):
        """
        data loader 的数据集相关的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--path', type=str, default=DATASET_DIR,
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml100k-1-5',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default=SEP,
                            help='sep of csv file.')
        parser.add_argument('--seq_sep', type=str, default=SEQ_SEP,
                            help='sep of sequences in csv file.')
        parser.add_argument('--label', type=str, default=LABEL,
                            help='name of dataset label column.')
        return parser

    def __init__(self, path, dataset, label=LABEL, load_data=True, sep=SEP, seq_sep=SEQ_SEP):
        """
        初始化
        :param path: 数据集目录
        :param dataset: 数据集名称
        :param label: 标签column的名称
        :param load_data: 是否要载入数据文件，否则只载入数据集信息
        :param sep: csv的分隔符
        :param seq_sep: 变长column的内部分隔符，比如历史记录可能为'1,2,4'
        """
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + TEST_SUFFIX)
        self.info_file = os.path.join(self.path, dataset + INFO_SUFFIX)
        self.user_file = os.path.join(self.path, dataset + USER_SUFFIX)
        self.item_file = os.path.join(self.path, dataset + ITEM_SUFFIX)
        self.train_pos_file = os.path.join(self.path, dataset + TRAIN_POS_SUFFIX)
        self.validation_pos_file = os.path.join(self.path, dataset + VALIDATION_POS_SUFFIX)
        self.test_pos_file = os.path.join(self.path, dataset + TEST_POS_SUFFIX)
        self.train_neg_file = os.path.join(self.path, dataset + TRAIN_NEG_SUFFIX)
        self.validation_neg_file = os.path.join(self.path, dataset + VALIDATION_NEG_SUFFIX)
        self.test_neg_file = os.path.join(self.path, dataset + TEST_NEG_SUFFIX)
        self.sep, self.seq_sep = sep, seq_sep
        self.load_data = load_data
        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self._load_user_item()
        self._load_data()
        self._load_his()
        self._load_info()
        if not os.path.exists(self.info_file) or self.load_data:
            self._save_info()

    def _load_user_item(self):
        """
        载入用户和物品的csv特征文件
        :return:
        """
        self.user_df, self.item_df = None, None
        if os.path.exists(self.user_file) and self.load_data:
            logging.info("load user csv...")
            self.user_df = pd.read_csv(self.user_file, sep='\t')
        if os.path.exists(self.item_file) and self.load_data:
            logging.info("load item csv...")
            self.item_df = pd.read_csv(self.item_file, sep='\t')

    def _load_data(self):
        """
        载入训练集、验证集、测试集csv文件
        :return:
        """
        if os.path.exists(self.train_file) and self.load_data:
            logging.info("load train csv...")
            self.train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("size of train: %d" % len(self.train_df))
            if self.label in self.train_df:
                logging.info("train label: " + str(dict(Counter(self.train_df[self.label]).most_common())))
        if os.path.exists(self.validation_file) and self.load_data:
            logging.info("load validation csv...")
            self.validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("size of validation: %d" % len(self.validation_df))
            if self.label in self.validation_df:
                logging.info("validation label: " + str(dict(Counter(self.validation_df[self.label]).most_common())))
        if os.path.exists(self.test_file) and self.load_data:
            logging.info("load test csv...")
            self.test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("size of test: %d" % len(self.test_df))
            if self.label in self.test_df:
                # logging.info("test label: " + str(dict(Counter(self.test_df[self.label]))))
                logging.info("test label: " + str(dict(Counter(self.test_df[self.label]).most_common())))

    def _save_info(self):

        def json_type(o):
            if isinstance(o, np.int64):
                return int(o)
            # if isinstance(o, np.float32): return int(o)
            raise TypeError

        max_json = json.dumps(self.column_max, default=json_type)
        min_json = json.dumps(self.column_min, default=json_type)
        out_f = open(self.info_file, 'w')
        out_f.write(max_json + os.linesep + min_json)
        logging.info('Save dataset info to ' + self.info_file)

    def _load_info(self):
        """
        载入数据集信息文件，如果不存在则创建
        :return:
        """
        max_dict, min_dict = {}, {}
        if not os.path.exists(self.info_file) or self.load_data:
            for df in [self.train_df, self.validation_df, self.test_df, self.user_df, self.item_df]:
                if df is None:
                    continue
                for c in df.columns:
                    if c not in max_dict:
                        max_dict[c] = df[c].max()
                    else:
                        max_dict[c] = max(df[c].max(), max_dict[c])
                    if c not in min_dict:
                        min_dict[c] = df[c].min()
                    else:
                        min_dict[c] = min(df[c].min(), min_dict[c])
        else:
            lines = open(self.info_file, 'r').readlines()
            max_dict = json.loads(lines[0])
            min_dict = json.loads(lines[1])

        self.column_max = max_dict
        self.column_min = min_dict

        # label的最小值和最大值
        self.label_max = self.column_max[self.label]
        self.label_min = self.column_min[self.label]
        logging.info("label: %d-%d" % (self.label_min, self.label_max))

        # 用户数目、物品数目
        self.user_num, self.item_num = 0, 0
        if UID in self.column_max:
            self.user_num = self.column_max[UID] + 1
        if IID in self.column_max:
            self.item_num = self.column_max[IID] + 1
        logging.info("# of users: %d" % self.user_num)
        logging.info("# of items: %d" % self.item_num)

        # 数据集的特征数目
        self.user_features = [f for f in self.column_max.keys() if f.startswith('u_')]
        logging.info("# of user features: %d" % len(self.user_features))
        self.item_features = [f for f in self.column_max.keys() if f.startswith('i_')]
        logging.info("# of item features: %d" % len(self.item_features))
        self.context_features = [f for f in self.column_max.keys() if f.startswith('c_')]
        logging.info("# of context features: %d" % len(self.context_features))
        self.features = self.context_features + self.user_features + self.item_features
        logging.info("# of features: %d" % len(self.features))

    def _load_his(self):
        """
        载入数据集按uid合并的历史交互记录，两列 'uid' 和 'iids'，没有则创建
        :return:
        """
        if not self.load_data or UID not in self.train_df or IID not in self.train_df:
            return
        if not os.path.exists(self.train_pos_file):
            logging.info("building train pos history csv...")
            train_pos_df = group_user_interactions_df(self.train_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            train_pos_df.to_csv(self.train_pos_file, index=False, sep=self.sep)
        if not os.path.exists(self.validation_pos_file):
            logging.info("building validation pos history csv...")
            validation_pos_df = group_user_interactions_df(
                self.validation_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            validation_pos_df.to_csv(self.validation_pos_file, index=False, sep=self.sep)
        if not os.path.exists(self.test_pos_file):
            logging.info("building test pos history csv...")
            test_pos_df = group_user_interactions_df(self.test_df, pos_neg=1, label=self.label, seq_sep=self.seq_sep)
            test_pos_df.to_csv(self.test_pos_file, index=False, sep=self.sep)

        if not os.path.exists(self.train_neg_file):
            logging.info("building train neg history csv...")
            train_neg_df = group_user_interactions_df(self.train_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            train_neg_df.to_csv(self.train_neg_file, index=False, sep=self.sep)
        if not os.path.exists(self.validation_neg_file):
            logging.info("building validation neg history csv...")
            validation_neg_df = group_user_interactions_df(
                self.validation_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            validation_neg_df.to_csv(self.validation_neg_file, index=False, sep=self.sep)
        if not os.path.exists(self.test_neg_file):
            logging.info("building test neg history csv...")
            test_neg_df = group_user_interactions_df(self.test_df, pos_neg=0, label=self.label, seq_sep=self.seq_sep)
            test_neg_df.to_csv(self.test_neg_file, index=False, sep=self.sep)

        def build_his(his_df, seqs_sep):
            uids = his_df[UID].tolist()
            iids = his_df[IIDS].astype(str).str.split(seqs_sep).values
            # iids = [i.split(self.seq_sep) for i in his_df['iids'].tolist()]
            iids = [[int(j) for j in i] for i in iids]
            user_his = dict(zip(uids, iids))
            return user_his

        self.train_pos_df, self.train_user_pos = None, None
        self.validation_pos_df, self.validation_user_pos = None, None
        self.test_pos_df, self.test_user_pos = None, None
        self.train_neg_df, self.train_user_neg = None, None
        self.validation_neg_df, self.validation_user_neg = None, None
        self.test_neg_df, self.test_user_neg = None, None
        if self.load_data:
            logging.info("load history csv...")
            self.train_pos_df = pd.read_csv(self.train_pos_file, sep=self.sep)
            self.train_user_pos = build_his(self.train_pos_df, self.seq_sep)
            self.validation_pos_df = pd.read_csv(self.validation_pos_file, sep=self.sep)
            self.validation_user_pos = build_his(self.validation_pos_df, self.seq_sep)
            self.test_pos_df = pd.read_csv(self.test_pos_file, sep=self.sep)
            self.test_user_pos = build_his(self.test_pos_df, self.seq_sep)
            self.train_neg_df = pd.read_csv(self.train_neg_file, sep=self.sep)
            self.train_user_neg = build_his(self.train_neg_df, self.seq_sep)
            self.validation_neg_df = pd.read_csv(self.validation_neg_file, sep=self.sep)
            self.validation_user_neg = build_his(self.validation_neg_df, self.seq_sep)
            self.test_neg_df = pd.read_csv(self.test_neg_file, sep=self.sep)
            self.test_user_neg = build_his(self.test_neg_df, self.seq_sep)

    def feature_info(self, include_id=True, include_item_features=True, include_user_features=True,
                     include_context_features=True):
        """
        生成模型需要的特征数目、维度等信息，特征最终会在DataProcesso中r转换为multi-hot的稀疏标示，
        例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
        那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
        :param include_id: 模型是否将uid,iid当成普通特征看待，将和其他特征一起转换到multi-hot embedding中，否则是单独两列
        :param include_item_features: 模型是否包含物品特征
        :param include_user_features: 模型是否包含用户特征
        :param include_context_features: 模型是否包含上下文特征
        :return: 所有特征，例['uid', 'iid', 'u_age', 'i_xx']
                 所有特征multi-hot之后的总维度，例 11
                 每个特征在mult-hot中所在范围的最小index，例[0, 3, 6, 9]
                 每个特征在mult-hot中所在范围的最大index，例[2, 5, 8, 10]
        """
        features = []
        if include_id:
            if UID in self.column_max:
                features.append(UID)
            if IID in self.column_max:
                features.append(IID)
        if include_user_features:
            features.extend(self.user_features)
        if include_item_features:
            features.extend(self.item_features)
        if include_context_features:
            features.extend(self.context_features)
        feature_dims = 0
        feature_min, feature_max = [], []
        for f in features:
            feature_min.append(feature_dims)
            feature_dims += int(self.column_max[f] + 1)
            feature_max.append(feature_dims - 1)
        logging.info('Model # of features %d' % len(features))
        logging.info('Model # of feature dims %d' % feature_dims)
        return features, feature_dims, feature_min, feature_max

    def append_his(self, last_n=10, supply=True, neg=False, neg_column=True):
        """
        根据训练集、验证集、测试集生成每条交互所对应的当时的历史。
        前提是要求数据集train,validation,test按时间排序，且train早于validation早于test。
        例如用户A正向交互是 点1, 没点2, 点3: 若supply=False
        neg=False时，'history' column对应的三条记录为['None', '1', '1']
        neg=True,neg_column=True时，
            'history' column对应的三条记录为['None', '1', '1']，'history_neg' column对应的三条记录为['None', 'None', '2']
        neg=True,neg_column=False时，'history' column对应的三条记录为['None', '1', '1,~2']
        :param last_n: 最长保留几个交互记录，<=0表示全部保留
        :param supply: 不够last_n的是否补全为-1
        :param neg: 历史记录是否包括负反馈（没点、没买的记录）
        :param neg_column: 负反馈历史记录是否单独一个column，当且仅当neg_include=True时有效
        :return:
        """
        his_dict, neg_dict = {}, {}
        for df in [self.train_df, self.validation_df, self.test_df]:
            if df is None:
                continue
            history, neg_history = [], []
            uids, iids, labels = df[UID].tolist(), df[IID].tolist(), df[self.label].tolist()
            for i, uid in enumerate(uids):
                iid, label = str(iids[i]), labels[i]
                if uid not in his_dict:
                    his_dict[uid] = []
                if uid not in neg_dict:
                    neg_dict[uid] = []

                tmp_his = his_dict[uid] if last_n <= 0 else his_dict[uid][-last_n:]
                tmp_neg = neg_dict[uid] if last_n <= 0 else neg_dict[uid][-last_n:]
                if supply:
                    tmp_his = tmp_his + ['-1'] * last_n
                    tmp_neg = tmp_neg + ['-1'] * last_n
                history.append(SEQ_SEP.join(tmp_his[:last_n]))
                neg_history.append(SEQ_SEP.join(tmp_neg[:last_n]))
                # if uid == 259 and iid == '928':
                #     print(his_dict[uid])
                # if uid == 259 and iid == '288':
                #     print(label)
                if label <= 0 and not neg_column and neg:
                    his_dict[uid].append('~' + iid)
                elif label <= 0 and neg_column:
                    neg_dict[uid].append(iid)
                elif label > 0:
                    his_dict[uid].append(iid)
            df[C_HISTORY] = history
            if neg and neg_column:
                df[C_HISTORY_NEG] = neg_history
        # print(self.train_df[self.train_df['uid'] == 3])
        # print(self.validation_df[self.validation_df['uid'] == 3])
        # print(self.test_df[self.test_df['uid'] == 3])
        # assert 1 == 2

    def drop_neg(self, train=True, validation=True, test=True):
        """
        如果是top n推荐，只保留正例，负例是训练过程中采样得到
        :return:
        """
        logging.info('Drop Neg Samples...')
        if train and self.train_df is not None:
            self.train_df = self.train_df[self.train_df[self.label] > 0].reset_index(drop=True)
        if validation and self.validation_df is not None:
            self.validation_df = self.validation_df[self.validation_df[self.label] > 0].reset_index(drop=True)
        if test and self.test_df is not None:
            self.test_df = self.test_df[self.test_df[self.label] > 0].reset_index(drop=True)
        logging.info("size of train: %d" % len(self.train_df))
        logging.info("size of validation: %d" % len(self.validation_df))
        logging.info("size of test: %d" % len(self.test_df))

    def label_01(self, train=True, validation=True, test=True):
        """
        讲label转换为01二值
        :return:
        """
        logging.info("Transform label to 0-1")
        if train and self.train_df is not None and self.label in self.train_df:
            self.train_df[self.label] = self.train_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("train label: " + str(dict(Counter(self.train_df[self.label]).most_common())))
        if validation and self.validation_df is not None and self.label in self.validation_df:
            self.validation_df[self.label] = self.validation_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("validation label: " + str(dict(Counter(self.validation_df[self.label]).most_common())))
        if test and self.test_df is not None and self.label in self.test_df:
            self.test_df[self.label] = self.test_df[self.label].apply(lambda x: 1 if x > 0 else 0)
            logging.info("test label: " + str(dict(Counter(self.test_df[self.label]).most_common())))
        self.label_min = 0
        self.label_max = 1
