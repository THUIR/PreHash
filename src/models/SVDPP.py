# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *


class SVDPP(RecModel):
    data_processor = 'HistoryDP'  # 默认data_processor

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.iid_embeddings_implicit = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'iid_embeddings_implicit', 'user_bias', 'item_bias']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]
        history = feed_dict[C_HISTORY]
        if 'sparse' in str(history.type()):
            his_vector = history.mm(self.iid_embeddings_implicit.weight)
        else:
            valid_his = history.gt(0).long()  # Batch * His
            his_vector = self.iid_embeddings_implicit(history * valid_his).sum(dim=1)

        # normalize
        his_length = feed_dict[C_HISTORY_LENGTH]
        valid_his = his_length.gt(0).float()
        tmp_length = his_length.float() * valid_his + (1 - valid_his) * 1
        his_vector = his_vector / tmp_length.sqrt().view([-1, 1])

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        # check_list.append(('cf_u_vectors', cf_u_vectors))
        # check_list.append(('his_vector', his_vector))
        embedding_l2.extend([u_bias, i_bias, cf_u_vectors, cf_i_vectors])

        prediction = ((cf_u_vectors + his_vector) * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        # prediction = prediction + self.global_bias
        # check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
