# coding=utf-8

import torch
import torch.nn.functional as F
from models.DeepModel import DeepModel
from utils import utils
from utils.global_p import *


class WideDeep(DeepModel):
    def _init_weights(self):
        self.feature_embeddings = torch.nn.Embedding(self.feature_dims, self.f_vector_size)
        self.cross_bias = torch.nn.Embedding(self.feature_dims, 1)
        self.l2_embeddings = ['feature_embeddings', 'cross_bias']

        pre_size = self.f_vector_size * self.feature_num
        for i, layer_size in enumerate(self.layers):
            setattr(self, 'layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
            setattr(self, 'bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            pre_size = layer_size
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        nonzero_embeddings = self.feature_embeddings(feed_dict[X])
        embedding_l2.append(nonzero_embeddings)
        pre_layer = nonzero_embeddings.view([-1, self.feature_num * self.f_vector_size])
        for i in range(0, len(self.layers)):
            pre_layer = getattr(self, 'layer_%d' % i)(pre_layer)
            pre_layer = getattr(self, 'bn_%d' % i)(pre_layer)
            pre_layer = F.relu(pre_layer)
            pre_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(pre_layer)
        deep_prediction = self.prediction(pre_layer).view([-1])
        cross_bias = self.cross_bias(feed_dict[X]).sum(dim=1).view([-1])
        embedding_l2.append(cross_bias)
        prediction = deep_prediction + cross_bias
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
