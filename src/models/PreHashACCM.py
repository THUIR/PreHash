# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
import numpy as np
from utils.global_p import *


class PreHashACCM(RecModel):
    include_id = False
    include_user_features = True
    include_item_features = True
    include_context_features = False
    data_loader = 'DataLoader'  # 默认data_loader
    data_processor = 'PreHashDP'  # 默认data_processor
    runner = 'BaseRunner'  # 默认runner

    @staticmethod
    def parse_model_args(parser, model_name='PreHashACCM'):
        parser.add_argument('--f_vector_size', type=int, default=64,
                            help='Size of feature vectors.')
        parser.add_argument('--cb_hidden_layers', type=str, default='[]',
                            help="Number of CB part's hidden layer.")
        parser.add_argument('--attention_size', type=int, default=16,
                            help='Size of attention layer.')
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        parser.add_argument('--hash_u_num', type=int, default=128,
                            help='Size of user hash.')
        parser.add_argument('--sample_max_n', type=int, default=128,
                            help='Sample top-n when learn hash.')
        parser.add_argument('--sample_r_n', type=int, default=128,
                            help='Sample random-n when learn hash.')
        parser.add_argument('--hash_layers', type=str, default='[32]',
                            help='MLP layer sizes of hash')
        parser.add_argument('--tree_layers', type=str, default='[64]',
                            help='Number of branches in each level of the hash tree')
        parser.add_argument('--transfer_att_size', type=int, default=16,
                            help='Size of attention layer of transfer layer (combine the hash and cf vector)')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, user_feature_num, item_feature_num, feature_dims,
                 f_vector_size, cb_hidden_layers, attention_size, cs_ratio,
                 hash_u_num, hash_layers, tree_layers, transfer_att_size,
                 sample_max_n, sample_r_n,
                 *args, **kwargs):
        self.user_feature_num = user_feature_num
        self.item_feature_num = item_feature_num
        self.feature_dims = feature_dims
        self.f_vector_size = f_vector_size
        self.cb_hidden_layers = cb_hidden_layers if type(cb_hidden_layers) == list else eval(cb_hidden_layers)
        self.attention_size = attention_size
        self.cs_ratio = cs_ratio
        self.hash_u_num = hash_u_num
        self.hash_layers = hash_layers if type(hash_layers) == list else eval(hash_layers)
        self.tree_layers = tree_layers if type(tree_layers) == list else eval(tree_layers)
        self.transfer_att_size = transfer_att_size
        self.sample_max_n, self.sample_r_n = sample_max_n, sample_r_n
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.hash_u_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.feature_embeddings = torch.nn.Embedding(self.feature_dims, self.f_vector_size)
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias', 'feature_embeddings']

        pre_size = self.i_vector_size
        for i, layer_size in enumerate(self.hash_layers):
            setattr(self, 'u_hash_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.u_hash_predict = torch.nn.Linear(pre_size, self.hash_u_num + sum(self.tree_layers), bias=False)
        self.transfer_att_layer = torch.nn.Linear(self.ui_vector_size, self.transfer_att_size)
        self.transfer_att_pre = torch.nn.Linear(self.transfer_att_size, 1, bias=False)

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
        i_bias = self.item_bias(i_ids).view([-1])
        cf_i_vectors = self.iid_embeddings(i_ids.view([-1, 1]))
        embedding_l2.extend([cf_i_vectors, i_bias])

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]

        # # history to hash_uid
        history = feed_dict[C_HISTORY]
        his_i_vectors = self.iid_embeddings(i_ids)
        if 'sparse' in str(history.type()):
            all_his_vector = history.mm(self.iid_embeddings.weight)
            if feed_dict[TRAIN]:
                # remove item i from history vectors
                if real_batch_size != total_batch_size:
                    padding_zeros = torch.zeros(size=[total_batch_size - real_batch_size, self.ui_vector_size],
                                                dtype=torch.float32)
                    padding_zeros = utils.tensor_to_gpu(padding_zeros)
                    tmp_his_i_vectors = torch.cat([his_i_vectors[:real_batch_size], padding_zeros])
                else:
                    tmp_his_i_vectors = his_i_vectors
                his_vector = all_his_vector - tmp_his_i_vectors
                his_length = feed_dict[C_HISTORY_LENGTH] - 1
            else:
                his_vector = all_his_vector
                his_length = feed_dict[C_HISTORY_LENGTH]
            embedding_l2.append(his_vector)
            # normalize alpha = 0.5
            valid_his = his_length.gt(0).float()
            tmp_length = his_length.float() * valid_his + (1 - valid_his) * 1
            his_vector = his_vector / tmp_length.sqrt().view([-1, 1])
        else:
            valid_his = history.gt(0).long()  # Batch * His
            if feed_dict[TRAIN]:
                if_target_item = (history != i_ids.view([-1, 1])).long()
                valid_his = if_target_item * valid_his
            his_length = valid_his.sum(dim=1, keepdim=True)
            his_vectors = self.iid_embeddings(history * valid_his)  # Batch * His * v
            valid_his = valid_his.view([total_batch_size, -1, 1]).float()  # Batch * His * 1
            his_vectors = his_vectors * valid_his  # Batch * His * v
            his_att = (his_vectors * cf_i_vectors).sum(dim=-1, keepdim=True).exp() * valid_his  # Batch * His * 1
            his_att_sum = his_att.sum(dim=1, keepdim=True)  # Batch * 1 * 1
            his_att_weight = his_att / (his_att_sum + 1e-8)
            all_his_vector = (his_vectors * his_att_weight).sum(dim=1)  # Batch * 64
            his_vector = all_his_vector
            embedding_l2.append(his_vector)
            # normalize alpha = 0.5
            his_vector = his_vector * his_length.float().sqrt().view([-1, 1])

        hash_layer = his_vector.detach()
        for i, layer_size in enumerate(self.hash_layers):
            hash_layer = getattr(self, 'u_hash_%d' % i)(hash_layer)
            hash_layer = F.relu(hash_layer)
            hash_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(hash_layer)

        # # tree hash
        u_tree_weights = self.u_hash_predict(hash_layer)
        u_tree_weights = u_tree_weights.clamp(min=-10)
        tree_layers = [0] + self.tree_layers + [self.hash_u_num]
        tree_layers_weights, lo, hi = [], 0, 0
        for i in range(len(tree_layers) - 1):
            lo, hi = lo + tree_layers[i], hi + tree_layers[i + 1]
            tree_layers_weights.append(u_tree_weights[:, lo:hi])
        u_hash_weights = tree_layers_weights[0].softmax(dim=-1)
        for i, weights in enumerate(tree_layers_weights[1:]):
            weights = weights.view([total_batch_size, tree_layers[i + 1], -1]).softmax(dim=-1)
            u_hash_weights = (weights * u_hash_weights.view([total_batch_size, -1, 1])).view([total_batch_size, -1])

        # check_list.append(('u_hash_weights_min', u_hash_weights.min(dim=1)[0].view([-1])))
        # check_list.append(('u_hash_weights_max', u_hash_weights.max(dim=1)[0].view([-1])))

        # # # get max prob hash id
        # u_max_prob_weights, u_max_prob_ids = u_hash_weights.topk(k=self.hash_u_num, dim=1, sorted=True)
        if not feed_dict[TRAIN]:
            sample_max_n = min(self.sample_max_n, self.hash_u_num)
            u_max_prob_weights, u_max_prob_ids = u_hash_weights.topk(k=sample_max_n, dim=1, sorted=True)
        else:
            sample_r_n = min(self.sample_r_n, self.hash_u_num)
            sample_uids = torch.randint(0, self.hash_u_num, size=[real_batch_size, sample_r_n]).long()
            sample_uids = utils.tensor_to_gpu(sample_uids)
            if real_batch_size != total_batch_size:
                sample_uids = torch.cat([sample_uids] * int(total_batch_size / real_batch_size))
            u_max_prob_weights, u_max_prob_ids = u_hash_weights.gather(1, sample_uids), sample_uids

        u_max_prob_weights = u_max_prob_weights / (u_max_prob_weights.sum(dim=-1, keepdim=True) + 1e-8)

        u_max_prob_vectors = self.uid_embeddings(u_max_prob_ids)
        u_max_prob_vectors = u_max_prob_vectors * u_max_prob_weights.unsqueeze(dim=2)
        u_max_prob_vectors = u_max_prob_vectors.sum(dim=1, keepdim=True)

        anchor_uids = feed_dict[K_ANCHOR_USER].view([-1, 1])
        if_anchor_uids = anchor_uids.gt(0).long()
        anchor_uids = anchor_uids * if_anchor_uids
        if_anchor_uids = if_anchor_uids.view([-1, 1, 1]).float()
        anchor_vectors = self.uid_embeddings(anchor_uids) * if_anchor_uids
        hash_anchor_vectors = anchor_vectors * if_anchor_uids + u_max_prob_vectors * (1 - if_anchor_uids)
        embedding_l2.append(hash_anchor_vectors)

        u_transfer_vectors = torch.cat((u_max_prob_vectors, his_vector.view_as(u_max_prob_vectors)), dim=1)
        if feed_dict[TRAIN] and 1 > self.cs_ratio > 0:
            drop_pos = torch.empty(size=(feed_dict[TOTAL_BATCH_SIZE], 2, 1)).bernoulli_(p=self.cs_ratio)
            random_vectors = torch.empty(size=u_transfer_vectors.size()).normal_(mean=0, std=0.01)
            drop_pos = utils.tensor_to_gpu(drop_pos)
            random_vectors = utils.tensor_to_gpu(random_vectors)
            u_transfer_vectors = u_transfer_vectors * (1 - drop_pos) + drop_pos * random_vectors
        u_transfer_att = self.transfer_att_pre(F.relu(self.transfer_att_layer(u_transfer_vectors))).softmax(dim=1)
        u_transfer_vectors = (u_transfer_vectors * u_transfer_att).sum(dim=1)
        # check_list.append(('u_transfer_vectors', u_transfer_vectors))

        cf_i_vectors = cf_i_vectors.view([-1, self.ui_vector_size])
        # cold sampling
        if feed_dict[TRAIN] and 1 > self.cs_ratio > 0:
            drop_ui_pos = utils.numpy_to_torch(
                np.random.choice(np.array([0, 1], dtype=np.float32), size=(feed_dict[TOTAL_BATCH_SIZE], 2),
                                 p=[1 - self.cs_ratio, self.cs_ratio]))
            # check_list.append(('drop_ui_pos', drop_ui_pos))
            drop_u_pos, drop_i_pos = drop_ui_pos[:, 0], drop_ui_pos[:, 1]
            drop_u_pos_v, drop_i_pos_v = drop_u_pos.view([-1, 1]), drop_i_pos.view([-1, 1])
            random_u_vectors = utils.numpy_to_torch(
                np.random.normal(0, 0.01, u_transfer_vectors.size()).astype(np.float32))
            random_i_vectors = utils.numpy_to_torch(np.random.normal(0, 0.01, cf_i_vectors.size()).astype(np.float32))
            i_bias = i_bias * (1 - drop_i_pos)
            u_transfer_vectors = random_u_vectors * drop_u_pos_v + u_transfer_vectors * (1 - drop_u_pos_v)
            cf_i_vectors = random_i_vectors * drop_i_pos_v + cf_i_vectors * (1 - drop_i_pos_v)

        # cf
        bias = i_bias + self.global_bias
        cf_prediction = (u_transfer_vectors * cf_i_vectors).sum(dim=-1).view([-1]) + bias

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
        ah_cf_u = self.attention_layer(u_transfer_vectors)
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

        u_vector = a_cf_u * u_transfer_vectors + a_cb_u * cb_u_vectors
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
