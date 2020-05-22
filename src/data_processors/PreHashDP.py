# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import math
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from data_processors.HistoryDP import HistoryDP
from utils.global_p import *

from pathos.multiprocessing import ProcessPool


class PreHashDP(HistoryDP):
    data_columns = [UID, IID, X, C_HISTORY, C_HISTORY_LENGTH]  # data dict中存储模型所需特征信息的key，需要转换为tensor
    info_columns = [SAMPLE_ID, TIME]  # data dict中存储模额外信息的key

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--select_anchor', type=int, default=1,
                            help='Whether select train anchor users.')
        parser.add_argument('--train_anchor', type=int, default=0,
                            help='Whether hash train anchor users.')
        parser.add_argument('--random_hash', type=int, default=0,
                            help='random hash all users.')
        parser.add_argument('--jaccard_hash', type=int, default=0,
                            help='select anchors and use jaccard similarity to match clusters.')
        parser.add_argument('--kmeans_hash', type=int, default=0,
                            help='use kmeans to form clusters and use cluster id as bucket id.')
        return HistoryDP.parse_dp_args(parser)

    @staticmethod
    def select_anchor_users(data_loader, anchor_num):
        anchor_users = [(u, len(data_loader.train_user_pos[u])) for u in data_loader.train_user_pos]
        anchor_users = sorted(anchor_users, key=lambda x: x[1], reverse=True)
        # np.random.shuffle(anchor_users)
        anchor_users = [u[0] for u in anchor_users[:anchor_num]]
        anchor_users = dict(zip(anchor_users, range(0, anchor_num)))
        return anchor_users

    @staticmethod
    def jaccard_hash_users(data_loader, anchor_num):
        anchor_users = [(u, len(data_loader.train_user_pos[u])) for u in data_loader.train_user_pos]
        anchor_users = sorted(anchor_users, key=lambda x: x[1], reverse=True)
        aus = [u[0] for u in anchor_users[:anchor_num]]
        anchor_users = dict(zip(aus, range(0, anchor_num)))
        users = list(data_loader.train_user_pos.keys())
        threads = 10
        thread_n = math.ceil(len(users) / threads)
        start_idx = [i * thread_n for i in range(threads)]
        end_idx = [i * thread_n + thread_n for i in range(threads)]

        def cluster(start, end, thread_no):
            for u in tqdm(users[start:end], leave=False, ncols=100, mininterval=1,
                          desc="jaccard_hash_users %d" % thread_no):
                if u in anchor_users:
                    continue
                max_jc, max_au = -1, -1
                for au in aus:
                    u_his, au_his = set(data_loader.train_user_pos[u]), set(data_loader.train_user_pos[au])
                    jc = 1.0 * len(u_his & au_his) / len(u_his | au_his)
                    if jc > max_jc:
                        max_jc, max_au = jc, au
                anchor_users[u] = anchor_users[max_au]
            return anchor_users

        pool = ProcessPool(nodes=threads)
        results = pool.map(cluster, start_idx, end_idx, range(1, threads + 1))
        for result in results:
            for u in result:
                anchor_users[u] = result[u]
        assert len(anchor_users) == len(users)
        return anchor_users

    @staticmethod
    def kmeans_hash_users(data_loader, anchor_num):
        anchor_users = [(u, data_loader.train_user_pos[u]) for u in data_loader.train_user_pos]
        documents = [' '.join([str(i) for i in u[1]]) for u in anchor_users]
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(documents)
        model = KMeans(n_clusters=anchor_num, init='k-means++', max_iter=100, n_init=1, verbose=1)
        prediction = model.fit_predict(x)
        result_dict = {}
        for i, u in enumerate(anchor_users):
            result_dict[u[0]] = prediction[i]
        assert len(result_dict) == len(anchor_users)
        return result_dict

    def __init__(self, select_anchor, train_anchor, random_hash, jaccard_hash, kmeans_hash,
                 hash_u_num, *args, **kwargs):
        HistoryDP.__init__(self, *args, **kwargs)

        self.anchor_users = None
        self.select_anchor = select_anchor
        self.train_anchor = train_anchor
        self.random_hash = random_hash
        self.hash_u_num = hash_u_num
        self.jaccard_hash = jaccard_hash
        self.kmeans_hash = kmeans_hash
        assert self.jaccard_hash + self.kmeans_hash < 2
        if self.random_hash == 1:
            self.anchor_users = {}
            for uid in range(self.data_loader.user_num):
                self.anchor_users[uid] = np.random.randint(self.hash_u_num)
        if self.select_anchor == 1:
            tmp_anchor_users = self.select_anchor_users(data_loader=self.data_loader, anchor_num=self.hash_u_num)
            if self.anchor_users is not None:
                for k in tmp_anchor_users:
                    self.anchor_users[k] = tmp_anchor_users[k]
            else:
                self.anchor_users = tmp_anchor_users
        if self.jaccard_hash == 1:
            self.anchor_users = self.jaccard_hash_users(data_loader=self.data_loader, anchor_num=self.hash_u_num)
        if self.kmeans_hash == 1:
            self.anchor_users = self.kmeans_hash_users(data_loader=self.data_loader, anchor_num=self.hash_u_num)

    def get_feed_dict(self, *args, **kwargs):
        feed_dict = HistoryDP.get_feed_dict(self, *args, **kwargs)

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        # anchor_users
        if self.anchor_users is not None:
            uids = list(feed_dict[UID].cpu().numpy())
            anchor_uids = [self.anchor_users[i] if i in self.anchor_users else -1 for i in uids]
            feed_dict[K_ANCHOR_USER] = utils.numpy_to_torch(np.array(anchor_uids, dtype=np.int64), gpu=False)
            # print(self.anchor_users)
            # print(feed_dict[K_ANCHOR_USER])
            # assert 1 == 2
        else:
            feed_dict[K_ANCHOR_USER] = utils.numpy_to_torch(-np.ones(total_batch_size, dtype=np.int64), gpu=False)
        return feed_dict
