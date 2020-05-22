# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from data_processors.DataProcessor import DataProcessor
from utils.global_p import *


class HistoryDP(DataProcessor):
    data_columns = [UID, IID, X, C_HISTORY, C_HISTORY_LENGTH]  # data dict中存储模型所需特征信息的key，需要转换为tensor
    info_columns = [SAMPLE_ID, TIME]  # data dict中存储模额外信息的key

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--max_his', type=int, default=-1,
                            help='Max history length.')
        parser.add_argument('--sup_his', type=int, default=0,
                            help='If sup_his > 0, supplement history list with -1 at the beginning')
        parser.add_argument('--sparse_his', type=int, default=1,
                            help='Whether use sparse representation of user history.')
        return DataProcessor.parse_dp_args(parser)

    def __init__(self, max_his, sup_his, sparse_his, *args, **kwargs):
        self.max_his = max_his
        self.sparse_his = sparse_his
        self.sup_his = sup_his
        DataProcessor.__init__(self, *args, **kwargs)

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
        feed_dict = DataProcessor.get_feed_dict(
            self, data, batch_start, batch_size, train, neg_data=neg_data,
            special_cols=[C_HISTORY] if special_cols is None else [C_HISTORY] + special_cols)
        # if train:
        #     print(feed_dict[C_HISTORY][batch_size - 10:batch_size])
        #     print(feed_dict[C_HISTORY][-10:])
        #     assert 1 == 2
        c, d = C_HISTORY, feed_dict[C_HISTORY]
        if self.sparse_his == 1:
            x, y = [], []
            for idx, iids in enumerate(d):
                iids = [iid for iid in iids if iid > 0]
                x.extend([idx] * len(iids))
                y.extend(iids)
            if len(x) <= 0:
                i = torch.LongTensor([[0], [0]])
                v = torch.FloatTensor([0.0])
            else:
                i = torch.LongTensor([x, y])
                v = torch.FloatTensor([1.0] * len(x))
            history = torch.sparse.FloatTensor(
                i, v, torch.Size([len(d), self.data_loader.item_num]))
            # if torch.cuda.device_count() > 0:
            #     history = history.cuda()
            feed_dict[c] = history
        else:
            max_length = max([len(x) for x in d])
            d = np.array([x + [-1] * (max_length - len(x)) for x in d])
            feed_dict[c] = utils.numpy_to_torch(d, gpu=False)
        return feed_dict

    # def prepare_batches(self, data, batch_size, train, model):
    #     """
    #     将data dict全部转换为batch
    #     :param data: dict 由self.get_*_data()和self.format_data_dict()系列函数产生
    #     :param batch_size: batch大小
    #     :param train: 训练还是测试
    #     :param model: Model类
    #     :return: list of batches
    #     """
    #     if self.sparse_his == 1 or self.sup_his == 1:
    #         return DataProcessor.prepare_batches(self, data=data, batch_size=batch_size, train=train, model=model)
    #
    #     buffer_key = ''
    #     if data is self.validation_data:
    #         buffer_key = '_'.join(['validation', str(batch_size), str(model)])
    #     elif data is self.test_data:
    #         buffer_key = '_'.join(['test', str(batch_size), str(model)])
    #     if buffer_key in self.vt_batches_buffer:
    #         return self.vt_batches_buffer[buffer_key]
    #
    #     if data is None:
    #         return None
    #     num_example = len(data[X])
    #     assert num_example > 0
    #     # 如果是训练，则需要对对应的所有正例采一个负例
    #     neg_data = None
    #     if train and self.rank == 1:
    #         neg_data = self.generate_neg_data(
    #             data, self.data_loader.train_df, sample_n=self.train_sample_n,
    #             train=True, model=model)
    #
    #     # 按历史记录长度聚合
    #     length_dict = {}
    #     lengths = [len(x) for x in data[C_HISTORY]]
    #     for idx, l in enumerate(lengths):
    #         if l not in length_dict:
    #             length_dict[l] = []
    #         length_dict[l].append(idx)
    #     lengths = list(length_dict.keys())
    #
    #     batches = []
    #     for l in tqdm(lengths, leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
    #         rows = length_dict[l]
    #         tmp_data = {}
    #         for key in data:
    #             tmp_data[key] = data[key][rows]
    #         tmp_neg_data = {} if train else None
    #         if train:
    #             for key in neg_data:
    #                 total_data_num = len(data[SAMPLE_ID])
    #                 tmp_neg_data[key] = neg_data[key][
    #                     np.concatenate([np.array(rows) + i * total_data_num for i in range(self.train_sample_n)])]
    #         tmp_total_batch = int((len(rows) + batch_size - 1) / batch_size)
    #         for batch in range(tmp_total_batch):
    #             batches.append(self.get_feed_dict(
    #                 tmp_data, batch * batch_size, batch_size, train, neg_data=tmp_neg_data))
    #     np.random.shuffle(batches)
    #     if buffer_key != '':
    #         self.vt_batches_buffer[buffer_key] = batches
    #     return batches

    def format_data_dict(self, df, model):
        """
        除了常规的uid,iid,label,user、item、context特征外，还需处理历史交互
        :param df: 训练、验证、测试df
        :param model: Model类
        :return:
        """

        if C_HISTORY in df:  # 如果已经由data_loader放在df里了，一般是动态seq形式
            history = df[[C_HISTORY]]
            history[C_HISTORY] = history[C_HISTORY].fillna('')
            his = history[C_HISTORY].apply(lambda x: eval('[' + x + ']'))
            if self.max_his > 0:
                his = his.apply(lambda x: x[-self.max_his:])
        else:  # 没有的话这里把训练集的补上
            uids = df[[UID]]
            user_his = self.data_loader.train_pos_df.copy().fillna('').rename(columns={IIDS: C_HISTORY})
            user_his[C_HISTORY] = user_his[C_HISTORY].apply(lambda x: eval('[' + x + ']'))
            if self.max_his > 0:
                user_his[C_HISTORY] = user_his[C_HISTORY].apply(lambda x: x[-self.max_his:])
            history = pd.merge(uids, user_his, on=UID, how='left')
            his = history[C_HISTORY]
            # print(his)

        his_length = his.apply(lambda x: len(x))
        if self.max_his > 0 and self.sup_his == 1:
            # 如果max_his > 0 self.sup_his==1，则从末尾截取max_his长度的历史，不够的在末尾补齐-1
            his = his.apply(lambda x: (x + [-1] * self.max_his)[:self.max_his])
        data_dict = DataProcessor.format_data_dict(self, df, model)
        data_dict[C_HISTORY] = his.values
        data_dict[C_HISTORY_LENGTH] = his_length.values
        # print(data_dict[C_HISTORY])
        return data_dict
