# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *


class FISM(RecModel):
    data_processor = 'HistoryDP'  # 默认data_processor

    def _init_weights(self):
        self.iid_embeddings_his = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['iid_embeddings_his', 'iid_embeddings', 'user_bias', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []

        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        real_batch_size = feed_dict[REAL_BATCH_SIZE]
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.append(cf_i_vectors)

        history = feed_dict[C_HISTORY]
        his_i_vectors = self.iid_embeddings_his(i_ids)
        if 'sparse' in str(history.type()):
            all_his_vector = history.mm(self.iid_embeddings_his.weight)
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
        else:
            valid_his = history.gt(0).long()  # Batch * His
            if feed_dict[TRAIN]:
                if_target_item = (history != i_ids.view([-1, 1])).long()
                valid_his = if_target_item * valid_his
            his_length = valid_his.sum(dim=1, keepdim=True)
            his_vectors = self.iid_embeddings_his(history * valid_his)  # Batch * His * v
            valid_his = valid_his.view([total_batch_size, -1, 1]).float()  # Batch * His * 1
            his_vectors = his_vectors * valid_his  # Batch * His * v
            his_vector = his_vectors.sum(dim=1)

        embedding_l2.append(his_vector)
        # normalize alpha = 0.5
        valid_his = his_length.gt(0).float()
        tmp_length = his_length.float() * valid_his + (1 - valid_his) * 1
        his_vector = his_vector / tmp_length.sqrt().view([-1, 1])

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])
        embedding_l2.extend([u_bias, i_bias])

        # cf_u_vectors = self.uid_embeddings(u_ids)
        # cf_i_vectors = self.iid_embeddings(i_ids)
        prediction = (his_vector * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        # prediction = prediction + self.global_bias
        # check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
