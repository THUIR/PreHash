# coding=utf-8
import pandas as pd
from collections import Counter, defaultdict
import os
import numpy as np
import socket
from shutil import copyfile
from utils.global_p import *


def group_user_interactions_csv(in_csv, out_csv, label=LABEL, sep=SEP):
    print('group_user_interactions_csv', out_csv)
    all_data = pd.read_csv(in_csv, sep=sep)
    group_inters = group_user_interactions_df(in_df=all_data, label=label)
    group_inters.to_csv(out_csv, sep=sep, index=False)
    return group_inters


def group_user_interactions_df(in_df, pos_neg, label=LABEL, seq_sep=SEQ_SEP):
    all_data = in_df
    if label in all_data.columns:
        if pos_neg == 1:
            all_data = all_data[all_data[label] > 0]
        elif pos_neg == 0:
            all_data = all_data[all_data[label] <= 0]
    uids, inters = [], []
    for name, group in all_data.groupby(UID):
        uids.append(name)
        inters.append(seq_sep.join(group[IID].astype(str).tolist()))
    group_inters = pd.DataFrame()
    group_inters[UID] = uids
    group_inters[IIDS] = inters
    return group_inters


def random_split_data(all_data_file, dataset_name, vt_ratio=0.1, copy_files=None, copy_suffixes=None):
    """
    随机切分已经生成的数据集文件 *.all.csv -> *.train.csv,*.validation.csv,*.test.csv
    :param all_data_file: 数据预处理完的文件 *.all.csv
    :param dataset_name: 给数据集起个名字
    :param vt_ratio: 验证集合测试集比例
    :param copy_files: 需要复制的文件
    :param copy_suffixes: 要复制的文件的生成的后缀名
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep=SEP)
    vt_size = int(len(all_data) * vt_ratio)
    validation_set = all_data.sample(n=vt_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=vt_size).sort_index()
    train_set = all_data.drop(test_set.index)
    # print(train_set)
    # print(validation_set)
    # print(test_set)
    train_set.to_csv(os.path.join(dir_name, dataset_name + TRAIN_SUFFIX), index=False, sep=SEP)
    validation_set.to_csv(os.path.join(dir_name, dataset_name + VALIDATION_SUFFIX), index=False, sep=SEP)
    test_set.to_csv(os.path.join(dir_name, dataset_name + TEST_SUFFIX), index=False, sep=SEP)

    # 复制用户、物品的特征文件
    if copy_files is not None:
        if type(copy_files) is str:
            copy_files = [copy_files]
        if type(copy_suffixes) is str:
            copy_suffixes = [copy_suffixes]
        assert (copy_suffixes is None or len(copy_files) == len(copy_suffixes))
        for i, copy_file in enumerate(copy_files):
            copyfile(copy_file, os.path.join(dir_name, dataset_name + copy_suffixes[i]))
    return train_set, validation_set, test_set


def leave_out_by_time_df(all_df, leave_n=1, warm_n=5, split_n=1, max_user=-1):
    min_label = all_df[LABEL].min()
    if min_label > 0:
        leave_df = all_df.groupby(UID).head(warm_n)
        all_df = all_df.drop(leave_df.index)
        split_dfs = []
        for i in range(split_n):
            total_uids = all_df[UID].unique()
            if 0 < max_user < len(total_uids):
                total_uids = np.random.choice(total_uids, size=max_user, replace=False).tolist()
                gb_uid = all_df.groupby(UID)
                split_df = []
                for uid in total_uids:
                    split_df.append(gb_uid.get_group(uid).tail(leave_n))
                split_df = pd.concat(split_df).sort_index()
            else:
                split_df = all_df.groupby(UID).tail(leave_n).sort_index()
            all_df = all_df.drop(split_df.index)
            split_dfs.append(split_df)
    else:
        leave_df = []
        for uid, group in all_df.groupby(UID):
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, LABEL] > 0:
                    found_idx = idx
                    found += 1
                    if found >= warm_n:
                        break
            if found > 0:
                leave_df.append(group.loc[:found_idx + 1])
        leave_df = pd.concat(leave_df)
        all_df = all_df.drop(leave_df.index)

        split_dfs = []
        for i in range(split_n):
            total_uids = all_df[all_df[LABEL] > 0][UID].unique()
            if 0 < max_user < len(total_uids):
                total_uids = np.random.choice(total_uids, size=max_user, replace=False).tolist()
            gb_uid = all_df.groupby(UID)
            split_df = []
            for uid in total_uids:
                group = gb_uid.get_group(uid)
                found, found_idx = 0, -1
                for idx in reversed(group.index):  # 倒着看该用户的历史，直到找到一个正样本
                    if group.loc[idx, LABEL] > 0:
                        found_idx = idx
                        found += 1
                        if found >= leave_n:
                            break
                if found > 0:  # 如果找到了正样本，则该样本及其之后的负样本全部放入测试集
                    split_df.append(group.loc[found_idx:])
            split_df = pd.concat(split_df).sort_index()
            all_df = all_df.drop(split_df.index)
            split_dfs.append(split_df)
    leave_df = pd.concat([leave_df, all_df]).sort_index()
    return leave_df, split_dfs


def leave_out_by_time_csv(all_data_file, dataset_name, leave_n=1, warm_n=5, u_f=None, i_f=None):
    """
    默认all_data里的交互是按时间顺序排列的，按交互顺序，把最后的交互划分到验证集合测试集里
    :param all_data_file: 数据预处理完的文件 *.all.csv，交互按时间顺序排列
    :param dataset_name: 给数据集起个名字
    :param leave_n: 验证和测试集保留几个用户交互
    :param warm_n: 保证测试用户在训练集中至少有warm_n个交互，否则交互全部放在训练集中
    :param u_f: 用户特征文件 *.user.csv
    :param i_f: 物品特征文件 *.item.csv
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('leave_out_by_time_csv', dir_name, leave_n, warm_n)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep=SEP)

    train_set, (test_set, validation_set) = leave_out_by_time_df(
        all_data, warm_n=warm_n, leave_n=leave_n, split_n=2, max_user=MAX_VT_USER)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))
    if UID in train_set.columns:
        print('train_user=%d validation_user=%d test_user=%d' %
              (len(train_set[UID].unique()), len(validation_set[UID].unique()), len(test_set[UID].unique())))

    train_set.to_csv(os.path.join(dir_name, dataset_name + TRAIN_SUFFIX), index=False, sep=SEP)
    validation_set.to_csv(os.path.join(dir_name, dataset_name + VALIDATION_SUFFIX), index=False, sep=SEP)
    test_set.to_csv(os.path.join(dir_name, dataset_name + TEST_SUFFIX), index=False, sep=SEP)
    # 复制用户、物品的特征文件
    if u_f is not None:
        copyfile(u_f, os.path.join(dir_name, dataset_name + USER_SUFFIX))
    if i_f is not None:
        copyfile(i_f, os.path.join(dir_name, dataset_name + ITEM_SUFFIX))
    return train_set, validation_set, test_set
