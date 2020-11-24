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
        #
        encoder_layers = TransformerEncoderLayer(
            d_model=self.feature_dim_size,
            nhead=1,
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
        output_transf = F.normalize(output_transf, p=2, dim=-1)
        #
        output_transf = output_transf.repeat(1, 1, self.num_neighbors)
        output_transf = output_transf.view(-1, self.feature_dim_size)
        #
        input_sampled_softmax = self.dropouts(output_transf)
        # r_embed=self.rel_feature(input_r)
        # r_sign=torch.sign(input_r)
        # r_embed*=r_sign[:,None]
        # r_embed = F.normalize(r_embed, p=2, dim=-1)
        # input_sampled_softmax+=r_embed
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
        node_embeddings = self.ss.weight.data.cpu().numpy()
        target_list = node_embeddings[range(self.vocab_size)]
        source_embed = node_embeddings[input_x]
        # relations
        r_embed = self.rel_feature(torch.tensor([input_r], device=device))
        r_sign = torch.sign(torch.tensor([input_r], device=device))
        r_embed *= r_sign[:, None]
        r_embed = F.normalize(r_embed, p=2, dim=-1).detach().cpu().numpy()
        # score
        target_list = target_list - (source_embed + r_embed)
        target_list = np.linalg.norm(target_list, axis=1)
        id_list = np.argpartition(target_list, 10)[:10]
        if input_t in id_list:
            return 1
        return 0
