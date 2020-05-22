# coding=utf-8

import torch
import logging
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils.rank_metrics import *
from utils.global_p import *
from utils import utils


class BaseModel(torch.nn.Module):
    """
    基类模型，一般新模型需要重载的函数有
    parse_model_args,
    __init__,
    _init_weights,
    predict,
    forward,
    """

    '''
    DataProcessor的format_data_dict()会用到这四个变量

    通常会把特征全部转换为multi-hot向量
    例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
    那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
    如果include_id=False，那么multi-hot不会包括uid,iid，即u_age=1,i_xx=0转化为010 10的稀疏表示 1,3
    include_user_features 和 include_item_features同理
    append id 是指是否将 uid,iid append在输入'X'的最前，比如在append_id=True, include_id=False的情况下：
    uid=0,iid=1,u_age=1,i_xx=0会转换为 0,1,1,3
    '''
    include_id = True
    include_user_features = True
    include_item_features = True
    include_context_features = True
    data_loader = 'DataLoader'  # 默认data_loader
    data_processor = 'DataProcessor'  # 默认data_processor
    runner = 'BaseRunner'  # 默认runner

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        """
        模型命令行参数
        :param parser:
        :param model_name: 模型名称
        :return:
        """
        parser.add_argument('--model_path', type=str,
                            default=os.path.join(MODEL_DIR, '%s/%s.pt' % (model_name, model_name)),
                            help='Model save path.')
        return parser

    @staticmethod
    def evaluate_method(p, data, metrics, error_skip=False):
        """
        计算模型评价指标
        :param p: 预测值，np.array，一般由runner.predict产生
        :param data: data dict，一般由DataProcessor产生
        :param metrics: 评价指标的list，一般是runner.metrics，例如 ['rmse', 'auc']
        :return:
        """
        l = data[Y]
        evaluations = []
        rank = False
        for metric in metrics:
            if '@' in metric:
                rank = True

        split_l, split_p, split_l_sum = None, None, None
        if rank:
            uids, times = data[UID].reshape([-1]), data[TIME].reshape([-1])
            if TIME in data:
                sorted_idx = np.lexsort((-l, -p, times, uids))
                sorted_uid, sorted_time = uids[sorted_idx], times[sorted_idx]
                sorted_key, sorted_spl = np.unique([sorted_uid, sorted_time], axis=1, return_index=True)
            else:
                sorted_idx = np.lexsort((-l, -p, uids))
                sorted_uid = uids[sorted_idx]
                sorted_key, sorted_spl = np.unique(sorted_uid, return_index=True)
            sorted_l, sorted_p = l[sorted_idx], p[sorted_idx]
            split_l, split_p = np.split(sorted_l, sorted_spl[1:]), np.split(sorted_p, sorted_spl[1:])
            split_l_sum = [np.sum((d > 0).astype(float)) for d in split_l]

        for metric in metrics:
            try:
                if metric == 'rmse':
                    evaluations.append(np.sqrt(mean_squared_error(l, p)))
                elif metric == 'mae':
                    evaluations.append(mean_absolute_error(l, p))
                elif metric == 'auc':
                    evaluations.append(roc_auc_score(l, p))
                elif metric == 'f1':
                    evaluations.append(f1_score(l, p))
                elif metric == 'accuracy':
                    evaluations.append(accuracy_score(l, np.around(p)))
                elif metric == 'precision':
                    evaluations.append(precision_score(l, p))
                elif metric == 'recall':
                    evaluations.append(recall_score(l, p))
                else:
                    k = int(metric.split('@')[-1])
                    if metric.startswith('ndcg@'):
                        max_k = max([len(d) for d in split_l])
                        k_data = np.array([(list(d) + [0] * max_k)[:max_k] for d in split_l])
                        best_rank = -np.sort(-k_data, axis=1)
                        best_dcg = np.sum(best_rank[:, :k] / np.log2(np.arange(2, k + 2)), axis=1)
                        best_dcg[best_dcg == 0] = 1
                        dcg = np.sum(k_data[:, :k] / np.log2(np.arange(2, k + 2)), axis=1)
                        ndcgs = dcg / best_dcg
                        evaluations.append(np.average(ndcgs))

                        # k_data = np.array([(list(d) + [0] * k)[:k] for d in split_l])
                        # best_rank = -np.sort(-k_data, axis=1)
                        # best_dcg = np.sum(best_rank / np.log2(np.arange(2, k + 2)), axis=1)
                        # best_dcg[best_dcg == 0] = 1
                        # dcg = np.sum(k_data / np.log2(np.arange(2, k + 2)), axis=1)
                        # ndcgs = dcg / best_dcg
                        # evaluations.append(np.average(ndcgs))
                    elif metric.startswith('hit@'):
                        k_data = np.array([(list(d) + [0] * k)[:k] for d in split_l])
                        hits = (np.sum((k_data > 0).astype(float), axis=1) > 0).astype(float)
                        evaluations.append(np.average(hits))
                    elif metric.startswith('precision@'):
                        k_data = [d[:k] for d in split_l]
                        k_data_dict = defaultdict(list)
                        for d in k_data:
                            k_data_dict[len(d)].append(d)
                        precisions = [np.average((np.array(d) > 0).astype(float), axis=1) for d in k_data_dict.values()]
                        evaluations.append(np.average(np.concatenate(precisions)))
                    elif metric.startswith('recall@'):
                        k_data = np.array([(list(d) + [0] * k)[:k] for d in split_l])
                        recalls = np.sum((k_data > 0).astype(float), axis=1) / split_l_sum
                        evaluations.append(np.average(recalls))
            except Exception as e:
                if error_skip:
                    evaluations.append(-1)
                else:
                    raise e
        return evaluations

    def __init__(self, label_min, label_max, feature_num, random_seed, model_path):
        super(BaseModel, self).__init__()
        self.label_min = label_min
        self.label_max = label_max
        self.feature_num = feature_num
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.model_path = model_path

        self._init_weights()
        logging.debug(list(self.parameters()))

        self.total_parameters = self.count_variables()
        logging.info('# of params: %d' % self.total_parameters)

        # optimizer 由runner生成并赋值
        self.optimizer = None

    def _init_weights(self):
        """
        初始化需要的权重（带权重层）
        :return:
        """
        self.x_bn = torch.nn.BatchNorm1d(self.feature_num)
        self.prediction = torch.nn.Linear(self.feature_num, 1)
        self.l2_embeddings = []

    def count_variables(self):
        """
        模型所有参数数目
        :return:
        """
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def init_paras(self, m):
        """
        模型自定义初始化函数，在main.py中会被调用
        :param m: 参数或含参数的层
        :return:
        """
        if 'Linear' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def l2(self, out_dict):
        """
        模型l2计算，默认是所有参数（除了embedding之外）的平方和，
        Embedding 的 L2是 只计算当前batch用到的
        :return:
        """
        l2 = utils.numpy_to_torch(np.array(0.0, dtype=np.float32), gpu=True)
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.split('.')[0] in self.l2_embeddings:
                continue
            l2 += (p ** 2).sum()
        for p in out_dict[EMBEDDING_L2]:
            l2 += (p ** 2).sum()
        return l2

    def predict(self, feed_dict):
        """
        只预测，不计算loss
        :param feed_dict: 模型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果
        """
        check_list = []
        x = self.x_bn(feed_dict[X].float())
        x = torch.nn.Dropout(p=feed_dict[DROPOUT])(x)
        prediction = F.relu(self.prediction(x)).view([-1])
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list}
        return out_dict

    def forward(self, feed_dict):
        """
        除了预测之外，还计算loss
        :param feed_dict: 型输入，是个dict
        :return: 输出，是个dict，prediction是预测值，check是需要检查的中间结果，loss是损失
        """
        out_dict = self.predict(feed_dict)
        if feed_dict[RANK] == 1:
            # 计算topn推荐的loss，batch前一半是正例，后一半是负例
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            # 计算rating/clicking预测的loss，默认使用mse
            loss = torch.nn.MSELoss()(out_dict[PREDICTION], feed_dict[Y])
        out_dict[LOSS] = loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict

    def rank_loss(self, prediction, label, real_batch_size):
        '''
        计算rank loss，类似BPR-max，参考论文:
        @inproceedings{hidasi2018recurrent,
          title={Recurrent neural networks with top-k gains for session-based recommendations},
          author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
          booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
          pages={843--852},
          year={2018},
          organization={ACM}
        }
        :param prediction: 预测值 [None]
        :param label: 标签 [None]
        :param real_batch_size: 观测值batch大小，不包括sample
        :return:
        '''
        pos_neg_tag = (label - 0.5) * 2
        observed, sample = prediction[:real_batch_size], prediction[real_batch_size:]
        # sample = sample.view([-1, real_batch_size]).mean(dim=0)
        sample = sample.view([-1, real_batch_size])
        sample_softmax = (sample * pos_neg_tag.view([1, real_batch_size])).softmax(dim=0)
        sample = (sample * sample_softmax).sum(dim=0)
        loss = -(pos_neg_tag * (observed - sample)).sigmoid().log().mean()
        return loss

    def lrp(self):
        pass

    def save_model(self, model_path=None):
        """
        保存模型，一般使用默认路径
        :param model_path: 指定模型保存路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None, cpu=False):
        """
        载入模型，一般使用默认路径
        :param model_path: 指定模型载入路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        if cpu:
            self.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)
