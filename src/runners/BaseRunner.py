# coding=utf-8

import torch
import logging
from time import time
from utils import utils
from utils.global_p import *
from tqdm import tqdm
import gc
import numpy as np
import copy
import os


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-6,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--grad_clip', type=float, default=10,
                            help='clip_grad_value_ para, -1 means, no clip')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metrics', type=str, default="ndcg@10,precision@1",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--pre_gpu', type=int, default=0,
                            help='Whether put all batches to gpu before run batches. \
                            If 0, dynamically put gpu for each batch.')
        return parser

    def __init__(self, optimizer='GD', lr=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, grad_clip=10, metrics='RMSE', check_epoch=10, early_stop=1, pre_gpu=0):
        """
        初始化
        :param optimizer: 优化器名字
        :param lr: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        """
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.grad_clip = grad_clip
        self.pre_gpu = pre_gpu

        # 把metrics转换为list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # 用来记录训练集、验证集、测试集每一轮的评价指标
        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_optimizer(self, model):
        """
        创建优化器
        :param model: 模型
        :return: 优化器
        """
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
            # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        return optimizer

    def _check_time(self, start=False):
        """
        记录时间用，self.time保存了[起始时间，上一步时间]
        :param start: 是否开始计时
        :return: 上一步到当前位置的时间
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        向所有batch添加一些控制信息比如DROPOUT
        :param batches: 所有batch的list，由DataProcessor产生
        :param train: 是否是训练阶段
        :return: 所有batch的list
        """
        for batch in batches:
            batch[TRAIN] = train
            batch[DROPOUT] = self.dropout if train else self.no_dropout
        return batches

    def predict(self, model, data, data_processor):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        """
        gc.collect()

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            prediction = model.predict(batch)[PREDICTION]
            predictions.append(prediction.detach().cpu().data.numpy())

        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])

        gc.collect()
        return predictions

    def fit(self, model, data, data_processor, epoch=-1):  # fit the results for an input set
        """
        训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :param epoch: 第几轮
        :return: 返回最后一轮的输出，可供self.check函数检查一些中间结果
        """
        gc.collect()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True, model=model)
        batches = self.batches_add_control(batches, train=True)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
        model.train()
        accumulate_size, prediction_list, output_dict = 0, [], None
        loss_list, loss_l2_list = [], []
        for i, batch in \
                tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            accumulate_size += len(batch[Y])
            model.optimizer.zero_grad()
            output_dict = model(batch)
            l2 = output_dict[LOSS_L2] * self.l2_weight
            loss = output_dict[LOSS] + l2
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())
            loss_l2_list.append(l2.detach().cpu().data.numpy())
            prediction_list.append(output_dict[PREDICTION].detach().cpu().data.numpy()[:batch[REAL_BATCH_SIZE]])
            if self.grad_clip > 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            if accumulate_size >= batch_size or i == len(batches) - 1:
                model.optimizer.step()
                accumulate_size = 0
            # model.optimizer.step()
        model.eval()
        gc.collect()

        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate([b[SAMPLE_ID][:b[REAL_BATCH_SIZE]] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        return predictions, output_dict, np.mean(loss_list), np.mean(loss_l2_list)

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        """
        metric = self.metrics[0]
        valid = self.valid_results
        # 如果已经训练超过20轮，且评价指标越小越好，且评价已经连续五轮非减
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-5:]):
            return True
        # 如果已经训练超过20轮，且评价指标越大越好，且评价已经连续五轮非增
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-5:]):
            return True
        # 训练好结果离当前已经20轮以上了
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False

    def train(self, model, data_processor):
        """
        训练模型
        :param model: 模型
        :param data_processor: DataProcessor实例
        :return:
        """

        # 获得训练、验证、测试数据，epoch=-1不shuffle
        train_data = data_processor.get_train_data(epoch=-1, model=model)
        validation_data = data_processor.get_validation_data(model=model)
        test_data = data_processor.get_test_data(model=model) if data_processor.unlabel_test == 0 else None
        self._check_time(start=True)  # 记录初始时间

        # 训练之前的模型效果
        init_train = self.evaluate(model, train_data, data_processor) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data, data_processor) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor) \
            if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))

        try:
            for epoch in range(self.epoch):
                self._check_time()
                # 每一轮需要重新获得训练数据，因为涉及shuffle或者topn推荐时需要重新采样负例
                epoch_train_data = data_processor.get_train_data(epoch=epoch, model=model)
                train_predictions, last_batch, mean_loss, mean_loss_l2 = \
                    self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                # 检查模型中间结果
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_loss_l2'] = mean_loss_l2
                    self.check(model, last_batch)
                training_time = self._check_time()

                # # evaluate模型效果
                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, metrics=['rmse'])
                valid_result = self.evaluate(model, validation_data, data_processor) \
                    if validation_data is not None else [-1.0] * len(self.metrics)
                test_result = self.evaluate(model, test_data, data_processor) \
                    if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
                testing_time = self._check_time()

                self.train_results.append(train_result)
                self.valid_results.append(valid_result)
                self.test_results.append(test_result)

                # 输出当前模型效果
                logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, utils.format_metric(train_result),
                                utils.format_metric(valid_result), utils.format_metric(test_result),
                                testing_time) + ','.join(self.metrics))

                # 如果当前效果是最优的，保存模型，基于验证集
                if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    model.save_model()
                # model.save_model(
                #     model_path='../model/variable_tsne_logic_epoch/variable_tsne_logic_epoch_%d.pt' % (epoch + 1))
                # 检查是否终止训练，基于验证集
                if self.eva_termination(model) and self.early_stop == 1:
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def evaluate(self, model, data, data_processor, metrics=None):  # evaluate the results for an input set
        """
        evaluate模型效果
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: list of float 每个对应一个 metric
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        检查模型中间结果
        :param model: 模型
        :param out_dict: 某一个batch的模型输出结果
        :return:
        """
        # batch = data_processor.get_feed_dict(data, 0, self.batch_size, True)
        # self.batches_add_control([batch], train=False)
        # model.eval()
        # check = model(batch)
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check[CHECK]):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['mean_loss'], check['mean_loss_l2']
        logging.info('mean loss = %.4f, l2 = %.4f' % (loss, l2))
        # if not (loss * 0.005 < l2 < loss * 0.1):
        #     logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))

    def run_some_tensors(self, model, data, data_processor, dict_keys):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        """
        gc.collect()

        if type(dict_keys) == str:
            dict_keys = [dict_keys]

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        result_dict = {}
        for key in dict_keys:
            result_dict[key] = []
        model.eval()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            out_dict = model.predict(batch)
            for key in dict_keys:
                if key in out_dict:
                    result_dict[key].append(out_dict[key].detach().cpu().data.numpy())

        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])
        for key in dict_keys:
            result_array = np.concatenate(result_dict[key])
            if len(sample_ids) == len(result_array):
                reorder_dict = dict(zip(sample_ids, result_array))
                result_dict[key] = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])

        gc.collect()
        return result_dict
