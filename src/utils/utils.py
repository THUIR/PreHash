# coding=utf-8
import logging
import numpy as np
import torch
from utils.global_p import *
import os
import inspect

LOWER_METRIC_LIST = ["rmse", 'mae']


def parse_global_args(parser):
    """
    全局命令行参数
    :param parser:
    :return:
    """
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default=os.path.join(LOG_DIR, 'log.txt'),
                        help='Logging file path')
    parser.add_argument('--result_file', type=str, default=os.path.join(RESULT_DIR, 'result.npy'),
                        help='Result file path')
    parser.add_argument('--random_seed', type=int, default=DEFAULT_SEED,
                        help='Random seed of numpy and tensorflow.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    return parser


def balance_data(data):
    """
    让正负样本数接近，正负样本数差距太大时使用
    :param data:
    :return:
    """
    pos_indexes = np.where(data['Y'] == 1)[0]
    copy_num = int((len(data['Y']) - len(pos_indexes)) / len(pos_indexes))
    if copy_num > 1:
        copy_indexes = np.tile(pos_indexes, copy_num)
        sample_index = np.concatenate([np.arange(0, len(data['Y'])), copy_indexes])
        for k in data:
            data[k] = data[k][sample_index]
    return data


def input_data_is_list(data):
    """
    如果data是一个dict的list，则合并这些dict，在测试多个数据集比如验证测试同时计算时
    :param data: dict or list
    :return:
    """
    if type(data) is list or type(data) is tuple:
        print("input_data_is_list")
        new_data = {}
        for k in data[0]:
            new_data[k] = np.concatenate([d[k] for d in data])
        return new_data
    return data


def format_metric(metric):
    """
    把计算出的评价指标转化为str，float保留四位小数
    :param metric:
    :return:
    """
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def shuffle_in_unison_scary(data):
    """
    shuffle整个数据集dict的内容
    :param data:
    :return:
    """
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    """
    求一个结果list中最佳的结果
    :param metric:
    :param results_list:
    :return:
    """
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def strictly_increasing(l):
    """
    判断是否严格单调增
    :param l:
    :return:
    """
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    """
    判断是否严格单调减
    :param l:
    :return:
    """
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    """
    判断是否单调非增
    :param l:
    :return:
    """
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    """
    判断是否单调非减
    :param l:
    :return:
    """
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    """
    判断是否单调
    :param l:
    :return:
    """
    return non_increasing(l) or non_decreasing(l)


def numpy_to_torch(d, gpu=True, requires_grad=True):
    """
    numpy array转化为pytorch tensor，有gpu则放到gpu
    :param d:
    :param gpu: whether put tensor to gpu
    :param requires_grad: whether the tensor requires grad
    :return:
    """
    t = torch.from_numpy(d)
    if d.dtype is np.float:
        t.requires_grad = requires_grad
    if gpu:
        t = tensor_to_gpu(t)
    return t


def tensor_to_gpu(t):
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict


def check_dir_and_mkdir(path):
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        print('make dirs:', dirname)
        os.makedirs(dirname)
    return
