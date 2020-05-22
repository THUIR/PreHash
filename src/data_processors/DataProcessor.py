# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
from utils.global_p import *


class DataProcessor(object):
    data_columns = [UID, IID, X]  # data dict中存储模型所需特征信息的key，需要转换为tensor
    info_columns = [SAMPLE_ID, TIME]  # data dict中存储模额外信息的key

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--test_sample_n', type=int, default=100,
                            help='Negative sample num for each instance in test/validation set when ranking.')
        parser.add_argument('--train_sample_n', type=int, default=1,
                            help='Negative sample num for each instance in train set when ranking.')
        parser.add_argument('--sample_un_p', type=float, default=1.0,
                            help='Sample from neg/pos with 1-p or unknown+neg/pos with p.')
        parser.add_argument('--unlabel_test', type=int, default=0,
                            help='If the label of test is unknown, do not sample neg of test set.')
        return parser

    @staticmethod
    def batch_to_gpu(batch):
        if torch.cuda.device_count() > 0:
            new_batch = {}
            for c in batch:
                if type(batch[c]) is torch.Tensor:
                    new_batch[c] = batch[c].cuda()
                else:
                    new_batch[c] = batch[c]
            return new_batch
        return batch

    def __init__(self, data_loader, rank, train_sample_n, test_sample_n, sample_un_p, unlabel_test=0):
        """
        初始化
        :param data_loader: DataLoader对象
        :param model: Model对象
        :param rank: 1=topn推荐 0=评分或点击预测
        :param test_sample_n: topn推荐时的测试集负例采样比例 正:负=1:test_sample_n
        """
        self.data_loader = data_loader
        self.rank = rank
        self.train_data, self.validation_data, self.test_data = None, None, None

        self.test_sample_n = test_sample_n
        self.train_sample_n = train_sample_n
        self.sample_un_p = sample_un_p
        self.unlabel_test = unlabel_test

        if self.rank == 1:
            # 生成用户交互的字典，方便采样负例时查询，不要采到正例
            self.train_history_pos = defaultdict(set)
            for uid in data_loader.train_user_pos.keys():
                self.train_history_pos[uid] = set(data_loader.train_user_pos[uid])
            self.validation_history_pos = defaultdict(set)
            for uid in data_loader.validation_user_pos.keys():
                self.validation_history_pos[uid] = set(data_loader.validation_user_pos[uid])
            self.test_history_pos = defaultdict(set)
            for uid in data_loader.test_user_pos.keys():
                self.test_history_pos[uid] = set(data_loader.test_user_pos[uid])

            self.train_history_neg = defaultdict(set)
            for uid in data_loader.train_user_neg.keys():
                self.train_history_neg[uid] = set(data_loader.train_user_neg[uid])
            self.validation_history_neg = defaultdict(set)
            for uid in data_loader.validation_user_neg.keys():
                self.validation_history_neg[uid] = set(data_loader.validation_user_neg[uid])
            self.test_history_neg = defaultdict(set)
            for uid in data_loader.test_user_neg.keys():
                self.test_history_neg[uid] = set(data_loader.test_user_neg[uid])
        self.vt_batches_buffer = {}

    def get_train_data(self, epoch, model):
        """
        将dataloader中的训练集Dataframe转换为所需要的字典后返回，每一轮都要shuffle
        该字典会被用来生成batches
        :param epoch: <0则不shuffle
        :param model: Model类
        :return: 字典dict
        """
        if self.train_data is None:
            logging.info('Prepare Train Data...')
            self.train_data = self.format_data_dict(self.data_loader.train_df, model)
            self.train_data[SAMPLE_ID] = np.arange(0, len(self.train_data[Y]))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data

    def get_validation_data(self, model):
        """
        将dataloader中的验证集Dataframe转换为所需要的字典后返回
        如果是topn推荐则每个正例对应采样test_sample_n个负例
        该字典会被用来生成batches
        :param model: Model类
        :return: 字典dict
        """
        if self.validation_data is None:
            logging.info('Prepare Validation Data...')
            df = self.data_loader.validation_df
            if self.rank == 1:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.validation_data = self.format_data_dict(df, model)
            self.validation_data[SAMPLE_ID] = np.arange(0, len(self.validation_data[Y]))
        return self.validation_data

    def get_test_data(self, model):
        """
        将dataloader中的测试集Dataframe转换为所需要的字典后返回
        如果是topn推荐则每个正例对应采样test_sample_n个负例
        该字典会被用来生成batches
        :param model: Model类
        :return: 字典dict
        """
        if self.test_data is None:
            logging.info('Prepare Test Data...')
            df = self.data_loader.test_df
            if self.rank == 1 and self.unlabel_test == 0:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.test_data = self.format_data_dict(df, model)
            self.test_data[SAMPLE_ID] = np.arange(0, len(self.test_data[Y]))
        return self.test_data

    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        """
        topn模型产生一个batch，如果是训练需要对每个正样本采样一个负样本，保证每个batch前一半是正样本，后一半是对应的负样本
        :param data: data dict，由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_start: batch开始的index
        :param batch_size: batch大小
        :param train: 训练还是测试
        :param neg_data: 负例的data dict，如果已经有可以传入拿来用
        :param special_cols: 需要特殊处理的column
        :return: batch的feed dict
        """
        # 如果是测试货验证，不需要对每个正样本采负样本，负样本已经1:test_sample_n采样好
        total_data_num = len(data[SAMPLE_ID])
        batch_end = min(len(data[self.data_columns[0]]), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        total_batch_size = real_batch_size * (self.train_sample_n + 1) if self.rank == 1 and train else real_batch_size
        feed_dict = {TRAIN: train, RANK: self.rank, REAL_BATCH_SIZE: real_batch_size,
                     TOTAL_BATCH_SIZE: total_batch_size}
        if Y in data:
            feed_dict[Y] = utils.numpy_to_torch(data[Y][batch_start:batch_start + real_batch_size], gpu=False)
        for c in self.info_columns + self.data_columns:
            if c not in data or data[c].size <= 0:
                continue
            d = data[c][batch_start: batch_start + real_batch_size]
            if self.rank == 1 and train:
                # print(c)
                # print(len(neg_data[c]))
                # print(total_data_num)
                # print(real_batch_size)
                neg_d = np.concatenate(
                    [neg_data[c][total_data_num * i + batch_start: total_data_num * i + batch_start + real_batch_size]
                     for i in range(self.train_sample_n)])
                d = np.concatenate([d, neg_d])
            feed_dict[c] = d
        for c in self.data_columns:
            if c not in feed_dict:
                continue
            if special_cols is not None and c in special_cols:
                continue
            # if feed_dict[c].dtype == np.object:
            #     d = np.array([x for x in feed_dict[c]])
            #     feed_dict[c] = utils.numpy_to_torch(d, gpu=False)
            # else:
            feed_dict[c] = utils.numpy_to_torch(feed_dict[c], gpu=False)
        return feed_dict

    def _check_vt_buffer(self, data, batch_size, model):
        buffer_key = ''
        if data is self.validation_data:
            buffer_key = '_'.join(['validation', str(batch_size), str(model)])
        elif data is self.test_data:
            buffer_key = '_'.join(['test', str(batch_size), str(model)])
        if buffer_key in self.vt_batches_buffer:
            return self.vt_batches_buffer[buffer_key]
        return buffer_key

    def prepare_batches(self, data, batch_size, train, model):
        """
        将data dict全部转换为batch
        :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生
        :param batch_size: batch大小
        :param train: 训练还是测试
        :param model: Model类
        :return: list of batches
        """

        buffer_key = self._check_vt_buffer(data=data, batch_size=batch_size, model=model)
        if type(buffer_key) != str:
            return buffer_key

        if data is None:
            return None
        num_example = len(data[X])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        # 如果是训练，则需要对对应的所有正例采一个负例
        neg_data = None
        if train and self.rank == 1:
            neg_data = self.generate_neg_data(
                data, self.data_loader.train_df, sample_n=self.train_sample_n,
                train=True, model=model)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(data=data, batch_start=batch * batch_size, batch_size=batch_size,
                                              train=train, neg_data=neg_data))

        if buffer_key != '':
            self.vt_batches_buffer[buffer_key] = batches
        return batches

    def format_data_dict(self, df, model):
        """
        将dataloader的训练、验证、测试Dataframe转换为需要的data dict
        :param df: pandas Dataframe, 在推荐问题中通常包含 UID,IID,'label' 三列
        :param model: Model类
        :return: data dict
        """

        data_loader = self.data_loader
        data = {}
        # 记录uid, iid
        out_columns = []
        if UID in df:
            out_columns.append(UID)
            data[UID] = df[UID].values
        if IID in df:
            out_columns.append(IID)
            data[IID] = df[IID].values
        if TIME in df:
            data[TIME] = df[TIME].values

        # label 记录在 Y 中
        if data_loader.label in df.columns:
            data[Y] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data[Y] = np.zeros(len(df), dtype=np.float32)

        ui_id = df[out_columns]

        # 根据uid和iid拼接上用户特征和物品特征
        out_df = ui_id
        if data_loader.user_df is not None and model.include_user_features:
            out_df = pd.merge(out_df, data_loader.user_df, on=UID, how='left')
        if data_loader.item_df is not None and model.include_item_features:
            out_df = pd.merge(out_df, data_loader.item_df, on=IID, how='left')

        # 是否包含context feature
        if model.include_context_features and len(data_loader.context_features) > 0:
            context = df[data_loader.context_features]
            out_df = pd.concat([out_df, context], axis=1, ignore_index=True)
        out_df = out_df.fillna(0)

        # 如果模型不把uid和iid当成普通特征一同看待，即不和其他特征一起转换为multi-hot向量
        if not model.include_id:
            out_df = out_df.drop(columns=out_columns)

        '''
        把特征全部转换为multi-hot向量
        例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
        那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
        '''
        base = 0
        for feature in out_df.columns:
            out_df[feature] = out_df[feature].apply(lambda x: x + base)
            base += int(data_loader.column_max[feature] + 1)

        # 如果模型需要，uid,iid拼接在x的前两列
        data[X] = out_df.values.astype(int)
        assert len(data[X]) == len(data[Y])
        return data

    def generate_neg_data(self, data, feature_df, sample_n, train, model):
        """
        产生neg_data dict一般为prepare_batches_rk train=True时所用
        :param data:
        :param feature_df:
        :param sample_n:
        :param train:
        :param model:
        :return:
        """
        inter_df = pd.DataFrame()
        for c in [UID, IID, Y, TIME]:
            if c in data:
                inter_df[c] = data[c]
            else:
                assert c == TIME
        neg_df = self.generate_neg_df(
            inter_df=inter_df, feature_df=feature_df,
            sample_n=sample_n, train=train)
        neg_data = self.format_data_dict(neg_df, model)
        neg_data[SAMPLE_ID] = np.arange(0, len(neg_data[Y])) + len(data[SAMPLE_ID])
        return neg_data

    def generate_neg_df(self, inter_df, feature_df, sample_n, train):
        """
        根据uid,iid和训练or验证测试的dataframe产生负样本
        :param sample_n: 负采样数目
        :param train: 训练集or验证集测试集负采样
        :return:
        """
        other_columns = [c for c in inter_df.columns if c not in [UID, Y]]
        neg_df = self._sample_neg_from_uid_list(
            uids=inter_df[UID].tolist(), labels=inter_df[Y].tolist(), sample_n=sample_n, train=train,
            other_infos=inter_df[other_columns].to_dict('list'))
        neg_df = pd.merge(neg_df, feature_df, on=[UID] + other_columns, how='left')
        neg_df = neg_df.drop(columns=[IID])
        neg_df = neg_df.rename(columns={'iid_neg': IID})
        neg_df = neg_df[feature_df.columns]
        neg_df[self.data_loader.label] = 0
        return neg_df

    def _sample_neg_from_uid_list(self, uids, labels, sample_n, train, other_infos=None):
        """
        根据uid的list采样对应的负样本
        :param uids: uid list
        :param sample_n: 每个uid采样几个负例
        :param train: 为训练集采样还是测试集采样
        :param other_infos: 除了uid,iid,label之外可能需要复制的信息，比如交互历史（前n个item），
            在generate_neg_df被用来复制原始正例iid
        :return: 返回DataFrame，还需经过self.format_data_dict()转换为data dict
        """
        if other_infos is None:
            other_infos = {}
        iid_list = []

        other_info_list = {}
        for info in other_infos:
            other_info_list[info] = []

        # 记录采样过程中采到的iid，避免重复采样
        item_num = self.data_loader.item_num
        for index, uid in enumerate(uids):
            if labels[index] > 0:
                # 避免采中已知的正例
                train_history = self.train_history_pos
                validation_history, test_history = self.validation_history_pos, self.test_history_pos
                known_train = self.train_history_neg
            else:
                assert train
                # 避免采中已知的负例
                train_history = self.train_history_neg
                validation_history, test_history = self.validation_history_neg, self.test_history_neg
                known_train = self.train_history_pos
            if train:
                # 训练集采样避免采训练集中已知的正例或负例
                inter_iids = train_history[uid]
            else:
                # 测试集采样避免所有已知的正例或负例
                inter_iids = train_history[uid] | validation_history[uid] | test_history[uid]

            # 检查所剩可以采样的数目
            remain_iids_num = item_num - len(inter_iids)
            # 所有可采数目不够则报错
            assert remain_iids_num >= sample_n

            # 如果数目不多则列出所有可采样的item采用np.choice
            remain_iids = None
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]

            sampled = set()
            if remain_iids is None:
                unknown_iid_list = []
                for i in range(sample_n):
                    iid = np.random.randint(1, self.data_loader.item_num)
                    while iid in inter_iids or iid in sampled:
                        iid = np.random.randint(1, self.data_loader.item_num)
                    unknown_iid_list.append(iid)
                    sampled.add(iid)
            else:
                unknown_iid_list = np.random.choice(remain_iids, sample_n, replace=False)

            # 如果训练时候，有可能从已知的负例或正例中采样
            if train and self.sample_un_p < 1:
                known_iid_list = list(np.random.choice(
                    list(known_train[uid]), min(sample_n, len(known_train[uid])), replace=False)) \
                    if len(known_train[uid]) != 0 else []
                known_iid_list = known_iid_list + unknown_iid_list
                tmp_iid_list = []
                sampled = set()
                for i in range(sample_n):
                    p = np.random.rand()
                    if p < self.sample_un_p or len(known_iid_list) == 0:
                        iid = unknown_iid_list.pop(0)
                        while iid in sampled:
                            iid = unknown_iid_list.pop(0)
                    else:
                        iid = known_iid_list.pop(0)
                        while iid in sampled:
                            iid = known_iid_list.pop(0)
                    tmp_iid_list.append(iid)
                    sampled.add(iid)
                iid_list.append(tmp_iid_list)
            else:
                iid_list.append(unknown_iid_list)

        all_uid_list, all_iid_list = [], []
        for i in range(sample_n):
            for index, uid in enumerate(uids):
                all_uid_list.append(uid)
                all_iid_list.append(iid_list[index][i])
                # # 复制其他信息
                for info in other_infos:
                    other_info_list[info].append(other_infos[info][index])

        neg_df = pd.DataFrame(data=list(zip(all_uid_list, all_iid_list)), columns=[UID, 'iid_neg'])
        for info in other_infos:
            neg_df[info] = other_info_list[info]
        return neg_df
