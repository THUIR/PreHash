# coding=utf-8

import torch
import torch.nn.functional as F
from models.BaseModel import BaseModel
from utils import utils
from utils.global_p import *


class RecModel(BaseModel):
    include_id = False
    include_user_features = False
    include_item_features = False
    include_context_features = False

    @staticmethod
    def parse_model_args(parser, model_name='RecModel'):
        parser.add_argument('--u_vector_size', type=int, default=64,
                            help='Size of user vectors.')
        parser.add_argument('--i_vector_size', type=int, default=64,
                            help='Size of item vectors.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, user_num, item_num, u_vector_size, i_vector_size,
                 *args, **kwargs):
        self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        self.user_num = user_num
        self.item_num = item_num
        BaseModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings']

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([cf_u_vectors, cf_i_vectors])

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        return out_dict
