# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
import numpy as np
from utils.global_p import *


class ACCM(RecModel):
    include_id = False
    include_user_features = True
    include_item_features = True
    include_context_features = False

    @staticmethod
    def parse_model_args(parser, model_name='ACCM'):
        parser.add_argument('--f_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--cb_hidden_layers', type=str, default='[]',
                            help="Number of CB part's hidden layer.")
        parser.add_argument('--attention_size', type=int, default=16,
                            help='Size of attention layer.')
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, user_feature_num, item_feature_num, feature_dims,
                 f_vector_size, cb_hidden_layers, attention_size, cs_ratio,
                 *args, **kwargs):
        self.user_feature_num = user_feature_num
        self.item_feature_num = item_feature_num
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.cb_hidden_layers = cb_hidden_layers if type(cb_hidden_layers) == list else eval(cb_hidden_layers)
        self.attention_size = attention_size
        self.cs_ratio = cs_ratio
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.feature_embeddings = torch.nn.Embedding(self.feature_dims, self.f_vector_size)
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias', 'feature_embeddings']

        user_pre_size = self.user_feature_num * self.f_vector_size
        item_pre_size = self.item_feature_num * self.f_vector_size
        layers = self.cb_hidden_layers + [self.ui_vector_size]
        for i, layer_size in enumerate(layers):
            setattr(self, 'user_layer_%d' % i, torch.nn.Linear(user_pre_size, layer_size))
            setattr(self, 'user_bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            user_pre_size = layer_size
            setattr(self, 'item_layer_%d' % i, torch.nn.Linear(item_pre_size, layer_size))
            setattr(self, 'item_bn_%d' % i, torch.nn.BatchNorm1d(layer_size))
            item_pre_size = layer_size

        self.attention_layer = torch.nn.Linear(self.ui_vector_size, self.attention_size)
        self.attention_prediction = torch.nn.Linear(self.attention_size, 1)

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])
        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([u_bias, i_bias, cf_u_vectors, cf_i_vectors])

        if feed_dict[TRAIN] and 1 > self.cs_ratio > 0:
            drop_ui_pos = utils.numpy_to_torch(
                np.random.choice(np.array([0, 1], dtype=np.float32), size=(feed_dict[TOTAL_BATCH_SIZE], 2),
                                 p=[1 - self.cs_ratio, self.cs_ratio]))
            # check_list.append(('drop_ui_pos', drop_ui_pos))
            drop_u_pos, drop_i_pos = drop_ui_pos[:, 0], drop_ui_pos[:, 1]
            drop_u_pos_v, drop_i_pos_v = drop_u_pos.view([-1, 1]), drop_i_pos.view([-1, 1])
            random_u_vectors = utils.numpy_to_torch(np.random.normal(0, 0.01, cf_u_vectors.size()).astype(np.float32))
            random_i_vectors = utils.numpy_to_torch(np.random.normal(0, 0.01, cf_i_vectors.size()).astype(np.float32))
            u_bias, i_bias = u_bias * (1 - drop_u_pos), i_bias * (1 - drop_i_pos)
            cf_u_vectors = random_u_vectors * drop_u_pos_v + cf_u_vectors * (1 - drop_u_pos_v)
            cf_i_vectors = random_i_vectors * drop_i_pos_v + cf_i_vectors * (1 - drop_i_pos_v)

        # cf
        bias = u_bias + i_bias + self.global_bias
        cf_prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1]) + bias

        # cb
        u_fs = feed_dict[X][:, :self.user_feature_num]
        i_fs = feed_dict[X][:, self.user_feature_num:]
        uf_layer = self.feature_embeddings(u_fs).view(-1, self.f_vector_size * self.user_feature_num)
        if_layer = self.feature_embeddings(i_fs).view(-1, self.f_vector_size * self.item_feature_num)
        embedding_l2.extend([uf_layer, if_layer])

        for i in range(0, len(self.cb_hidden_layers) + 1):
            uf_layer = getattr(self, 'user_layer_%d' % i)(uf_layer)
            if_layer = getattr(self, 'item_layer_%d' % i)(if_layer)
            uf_layer = getattr(self, 'user_bn_%d' % i)(uf_layer)
            if_layer = getattr(self, 'item_bn_%d' % i)(if_layer)
            if i < len(self.cb_hidden_layers):
                uf_layer = F.relu(uf_layer)
                uf_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(uf_layer)
                if_layer = F.relu(if_layer)
                if_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(if_layer)
        cb_u_vectors, cb_i_vectors = uf_layer, if_layer
        cb_prediction = (cb_u_vectors * if_layer).sum(dim=1).view([-1]) + bias

        # attention
        ah_cf_u = self.attention_layer(cf_u_vectors)
        ah_cf_u = torch.tanh(ah_cf_u)
        a_cf_u = self.attention_prediction(ah_cf_u)
        a_cf_u = torch.exp(a_cf_u)

        ah_cb_u = self.attention_layer(cb_u_vectors)
        ah_cb_u = torch.tanh(ah_cb_u)
        a_cb_u = self.attention_prediction(ah_cb_u)
        a_cb_u = torch.exp(a_cb_u)

        a_sum = a_cf_u + a_cb_u

        a_cf_u = a_cf_u / a_sum
        a_cb_u = a_cb_u / a_sum

        ah_cf_i = self.attention_layer(cf_i_vectors)
        ah_cf_i = torch.tanh(ah_cf_i)
        a_cf_i = self.attention_prediction(ah_cf_i)
        a_cf_i = torch.exp(a_cf_i)

        ah_cb_i = self.attention_layer(cb_i_vectors)
        ah_cb_i = torch.tanh(ah_cb_i)
        a_cb_i = self.attention_prediction(ah_cb_i)
        a_cb_i = torch.exp(a_cb_i)

        a_sum = a_cf_i + a_cb_i
        a_cf_i = a_cf_i / a_sum
        a_cb_i = a_cb_i / a_sum

        u_vector = a_cf_u * cf_u_vectors + a_cb_u * cb_u_vectors
        i_vector = a_cf_i * cf_i_vectors + a_cb_i * cb_i_vectors
        prediction = (u_vector * i_vector).sum(dim=1).view([-1]) + bias
        # check_list.append(('prediction', prediction))

        # cf_loss = torch.nn.MSELoss()(cf_prediction, feed_dict['Y'])
        # cb_loss = torch.nn.MSELoss()(cb_prediction, feed_dict['Y'])
        # loss = torch.nn.MSELoss()(prediction, feed_dict['Y']) + cf_loss + cb_loss
        out_dict = {PREDICTION: prediction,
                    'cb_prediction': cb_prediction, 'cf_prediction': cf_prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        if feed_dict[RANK] == 1:
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
            cf_loss = self.rank_loss(out_dict['cf_prediction'], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
            cb_loss = self.rank_loss(out_dict['cb_prediction'], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            loss = torch.nn.MSELoss()(out_dict[PREDICTION], feed_dict[Y])
            cf_loss = torch.nn.MSELoss()(out_dict['cf_prediction'], feed_dict[Y])
            cb_loss = torch.nn.MSELoss()(out_dict['cb_prediction'], feed_dict[Y])
        out_dict[LOSS] = loss + cf_loss + cb_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict
