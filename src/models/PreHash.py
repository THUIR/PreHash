# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
import numpy as np
from collections import defaultdict
import logging
from utils.global_p import *


class PreHash(RecModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False
    data_loader = 'DataLoader'  # 默认data_loader
    data_processor = 'PreHashDP'  # 默认data_processor
    runner = 'BaseRunner'  # 默认runner

    @staticmethod
    def parse_model_args(parser, model_name='PreHash'):
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
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, hash_u_num, hash_layers, tree_layers, transfer_att_size,
                 cs_ratio, sample_max_n, sample_r_n,
                 *args, **kwargs):
        self.hash_u_num = hash_u_num
        self.hash_layers = hash_layers if type(hash_layers) == list else eval(hash_layers)
        self.tree_layers = tree_layers if type(tree_layers) == list else eval(tree_layers)
        self.sample_max_n, self.sample_r_n = sample_max_n, sample_r_n
        self.transfer_att_size = transfer_att_size
        self.cs_ratio = cs_ratio

        self.rec_paras_record, self.hash_paras_record = {}, {}
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        pre_size = self.i_vector_size
        for i, layer_size in enumerate(self.hash_layers):
            setattr(self, 'u_hash_%d' % i, torch.nn.Linear(pre_size, layer_size))
            pre_size = layer_size
        self.u_hash_predict = torch.nn.Linear(pre_size, self.hash_u_num + sum(self.tree_layers), bias=False)

        if self.transfer_att_size > 0:
            self.transfer_att_layer = torch.nn.Linear(self.ui_vector_size, self.transfer_att_size)
            self.transfer_att_pre = torch.nn.Linear(self.transfer_att_size, 1, bias=False)

        self.uid_embeddings = torch.nn.Embedding(self.hash_u_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)

        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        i_bias = self.item_bias(i_ids).view([-1])
        cf_i_vectors = self.iid_embeddings(i_ids.view([-1, 1]))
        embedding_l2.extend([cf_i_vectors, i_bias])

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]
        # check_list.append(('u_ids', u_ids))

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
        his_prediction = (his_vector.view_as(cf_i_vectors) * cf_i_vectors).sum(dim=-1).view([-1])
        his_prediction = his_prediction + i_bias + self.global_bias

        if self.hash_u_num <= 0:
            out_dict = {PREDICTION: his_prediction, CHECK: check_list, EMBEDDING_L2: embedding_l2}
            return out_dict

        if self.transfer_att_size > 0:
            hash_layer = his_vector.detach()
        else:
            hash_layer = his_vector
        for i, layer_size in enumerate(self.hash_layers):
            hash_layer = getattr(self, 'u_hash_%d' % i)(hash_layer)
            hash_layer = F.relu(hash_layer)
            hash_layer = torch.nn.Dropout(p=feed_dict[DROPOUT])(hash_layer)

        # # tree hash
        u_tree_weights = self.u_hash_predict(hash_layer)
        # check_list.append(('u_tree_weights', u_tree_weights))
        u_tree_weights = u_tree_weights.clamp(min=-10)
        # check_list.append(('u_tree_weights', u_tree_weights))
        tree_layers = [0] + self.tree_layers + [self.hash_u_num]
        tree_layers_weights, lo, hi = [], 0, 0
        for i in range(len(tree_layers) - 1):
            lo, hi = lo + tree_layers[i], hi + tree_layers[i + 1]
            tree_layers_weights.append(u_tree_weights[:, lo:hi])
        u_hash_weights = tree_layers_weights[0].softmax(dim=-1)
        for i, weights in enumerate(tree_layers_weights[1:]):
            weights = weights.view([total_batch_size, tree_layers[i + 1], -1]).softmax(dim=-1)
            # print(u_hash_weights.size(), weights.size())
            u_hash_weights = (weights * u_hash_weights.view([total_batch_size, -1, 1])).view([total_batch_size, -1])
            # print(u_hash_weights.sum(dim=-1))

        u_ids_hash = torch.argmax(u_hash_weights, dim=1, keepdim=True)
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
            # sample_uids = utils.numpy_to_torch(
            #     np.random.randint(0, self.hash_u_num, size=[real_batch_size, self.sample_r_n], dtype=np.int))
            if real_batch_size != total_batch_size:
                sample_uids = torch.cat([sample_uids] * int(total_batch_size / real_batch_size))
            u_max_prob_weights, u_max_prob_ids = u_hash_weights.gather(1, sample_uids), sample_uids

        u_max_prob_weights = u_max_prob_weights / (u_max_prob_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # check_list.append(('u_max_prob_weights', u_max_prob_weights))
        # check_list.append(('u_max_prob_ids', u_max_prob_ids))
        u_max_prob_vectors = self.uid_embeddings(u_max_prob_ids)
        u_max_prob_vectors = u_max_prob_vectors * u_max_prob_weights.unsqueeze(dim=2)
        u_max_prob_vectors = u_max_prob_vectors.sum(dim=1, keepdim=True)

        anchor_uids = feed_dict[K_ANCHOR_USER].view([-1, 1])
        # check_list.append(('anchor_uids', anchor_uids))
        if_anchor_uids = anchor_uids.gt(0).long()
        anchor_uids = anchor_uids * if_anchor_uids
        if_anchor_uids = if_anchor_uids.view([-1, 1, 1]).float()
        anchor_vectors = self.uid_embeddings(anchor_uids) * if_anchor_uids
        # check_list.append(('anchor_vectors', anchor_vectors))
        hash_anchor_vectors = anchor_vectors * if_anchor_uids + u_max_prob_vectors * (1 - if_anchor_uids)
        # check_list.append(('hash_anchor_vectors', hash_anchor_vectors))
        embedding_l2.append(hash_anchor_vectors)

        hash_prediction = (hash_anchor_vectors * cf_i_vectors).sum(dim=-1).view([-1])
        # hash_prediction = (u_max_prob_vectors * cf_i_vectors).sum(dim=-1).view([-1])
        hash_prediction = hash_prediction + i_bias + self.global_bias
        if self.transfer_att_size <= 0:
            out_dict = {PREDICTION: hash_prediction, CHECK: check_list, EMBEDDING_L2: embedding_l2}
            return out_dict

        u_transfer_vectors = torch.cat((u_max_prob_vectors, his_vector.view_as(u_max_prob_vectors)), dim=1)
        if feed_dict[TRAIN] and 1 > self.cs_ratio > 0:
            drop_pos = torch.empty(size=(feed_dict[TOTAL_BATCH_SIZE], 2, 1)).bernoulli_(p=self.cs_ratio)
            random_vectors = torch.empty(size=u_transfer_vectors.size()).normal_(mean=0, std=0.01)
            drop_pos = utils.tensor_to_gpu(drop_pos)
            random_vectors = utils.tensor_to_gpu(random_vectors)
            # drop_pos = utils.numpy_to_torch(
            #     np.random.choice(np.array([0, 1], dtype=np.float32), size=(feed_dict[TOTAL_BATCH_SIZE], 2),
            #                      p=[1 - self.cs_ratio, self.cs_ratio]))
            # drop_pos = drop_pos.view([-1, 2, 1])
            # random_vectors = utils.numpy_to_torch(
            #     np.random.normal(0, 0.01, u_transfer_vectors.size()).astype(np.float32))
            u_transfer_vectors = u_transfer_vectors * (1 - drop_pos) + drop_pos * random_vectors
        u_transfer_att = self.transfer_att_pre(F.relu(self.transfer_att_layer(u_transfer_vectors))).softmax(dim=1)
        u_transfer_vectors = (u_transfer_vectors * u_transfer_att).sum(dim=1, keepdim=True)
        # u_transfer_vectors = his_vector.view_as(u_max_prob_vectors)
        # u_transfer_vectors = u_max_prob_vectors
        # u_transfer_vectors = hash_anchor_vectors

        prediction = (u_transfer_vectors * cf_i_vectors).sum(dim=2).view([-1])
        prediction = prediction + i_bias + self.global_bias

        out_dict = {PREDICTION: prediction, CHECK: check_list, EMBEDDING_L2: embedding_l2,
                    'his_prediction': his_prediction, 'hash_prediction': hash_prediction,
                    'u_hash_weights': u_hash_weights, 'u_ids_hash': u_ids_hash, 'anchor_uids': anchor_uids}
        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        if feed_dict[RANK] == 1:
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
            his_loss, hash_loss = 0, 0
            if self.hash_u_num > 0 and self.transfer_att_size > 0:
                his_loss = self.rank_loss(out_dict['his_prediction'], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
                hash_loss = self.rank_loss(out_dict['hash_prediction'], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            loss = torch.nn.MSELoss()(out_dict[PREDICTION], feed_dict[Y])
            his_loss, hash_loss = 0, 0
            if self.hash_u_num > 0 and self.transfer_att_size > 0:
                his_loss = torch.nn.MSELoss()(out_dict['his_prediction'], feed_dict[Y])
                hash_loss = torch.nn.MSELoss()(out_dict['hash_prediction'], feed_dict[Y])
        out_dict[LOSS] = loss + his_loss + hash_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict
