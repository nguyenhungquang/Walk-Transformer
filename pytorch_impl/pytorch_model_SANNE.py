import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sampled_softmax import *
import numpy as np

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class SANNE(nn.Module):
    def __init__(
        self,
        vocab_size,
        rel_size,
        feature_dim_size,
        ff_hidden_size,
        sampled_num,
        num_self_att_layers,
        dropout,
        device,
        num_heads,
        num_neighbors,
        initialization=None,
    ):
        super(SANNE, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_self_att_layers = num_self_att_layers
        self.vocab_size = vocab_size
        self.sampled_num = sampled_num
        self.device = device
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.rel_size = rel_size
        if initialization == None:
            self.input_feature = nn.Embedding(self.vocab_size, self.feature_dim_size)
            nn.init.xavier_uniform_(self.input_feature.weight.data)
        else:
            self.input_feature = nn.Embedding.from_pretrained(initialization)
        self.rel_feature = nn.Embedding(self.rel_size, self.feature_dim_size)
        # self.rel_feature = nn.Parameter(torch.Tensor(self.rel_size,self.feature_dim_size,self.feature_dim_size))
        #
        encoder_layers = TransformerEncoderLayer(
            d_model=self.feature_dim_size,
            nhead=num_heads,
            dim_feedforward=self.ff_hidden_size,
            dropout=0.5,
        )  # embed_dim must be divisible by num_heads
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, self.num_self_att_layers
        )
        # Linear function
        self.dropouts = nn.Dropout(dropout)
        self.ss = SampledSoftmax(
            self.vocab_size,
            self.rel_size,
            self.sampled_num,
            self.feature_dim_size,
            self.device,
        )

    def forward(self, input_x, input_r, input_y):
        #
        input_transf = self.input_feature(input_x)
        input_transf = F.normalize(input_transf, p=2, dim=-1)
        output_transf = self.transformer_encoder(input_transf)
        input_transf = F.normalize(output_transf, p=2, dim=-1)
        # #
        input_transf = input_transf.repeat(1, 1, self.num_neighbors)
        output_transf = input_transf.view(-1, self.feature_dim_size)
        # # 
        input_sampled_softmax = self.dropouts(output_transf)

        #relations
        #distmult matrix
        # r_embed=torch.index_select(self.rel_feature,0,abs(input_r))
        # r_sign=torch.sign(input_r)
        # r_neg_index=torch.nonzero(r_sign<0)
        # if r_neg_index.nelement() != 0 :
        #     r_neg_emb=torch.index_select(r_embed,0,r_neg_index).permute(0,2,1)
        #     r_embed[r_neg_index,:,:]=r_neg_emb

        #distmult vector
        # r_embed = self.rel_feature(abs(input_r))
        # input_sampled_softmax=torch.mul(input_sampled_softmax,r_embed)

        #transe
        r_embed=self.rel_feature(abs(input_r))
        r_sign=torch.sign(input_r)
        r_embed*=r_sign[:,None]
        input_sampled_softmax+=r_embed

        # input_sampled_softmax=torch.matmul(input_sampled_softmax.view(-1,1,self.feature_dim_size),r_embed).view(-1,self.feature_dim_size)
        # input_sampled_softmax=F.normalize(input_sampled_softmax,p=2,dim=-1)
        logits = self.ss(input_sampled_softmax, input_y)

        return logits

    def predict(self, input_x):
        #
        input_transf = self.input_feature(input_x)
        input_transf = F.normalize(input_transf, p=2, dim=-1)
        output_transf = self.transformer_encoder(input_transf)
        # output_transf = F.normalize(output_transf, p=2, dim=-1) # keep ???

        return output_transf

    def hit_at_10(self, input_x, input_r, input_t, device="cpu"):
        node_embeddings = self.ss.weight.data#.cpu().numpy()
        target_list = torch.index_select(node_embeddings,0,torch.tensor(range(self.vocab_size),device=device))
        source_embed = torch.index_select(node_embeddings,0,torch.tensor(input_x,device=device))
        # relations
        # r_embed = self.rel_feature(torch.tensor([abs(input_r)],device=device))

        # r_embed=torch.index_select(self.rel_feature,0,torch.tensor([abs(input_r)],device=device))
        # # score
        # source_embed=torch.matmul(source_embed,r_embed)
        # source_embed=F.normalize(source_embed)
        # target_list = torch.matmul(target_list,source_embed.view(-1,1)).flatten().detach().cpu().numpy()
        # # target_list = np.linalg.norm(target_list, axis=1)
        # id_list = np.argsort(target_list)[-10:]
        # if input_t in id_list:
        #     return 1
        # return 0

        r_embed = self.rel_feature(torch.tensor([input_r], device=device))
        # score
        target_list = target_list - (source_embed + r_embed)
        target_list = np.linalg.norm(target_list.detach().cpu().numpy(), axis=1)
        id_list = np.argpartition(target_list, 10)[:10]
        if input_t in id_list:
            return 1
        return 0
