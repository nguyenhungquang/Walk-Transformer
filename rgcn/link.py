"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR
  probably because the model only uses one GNN layer so messages are propagated
  among immediate neighbors. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""
from tqdm.auto import tqdm
import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.sparse as scipy
from dgl.data.knowledge_graph import load_data
from dgl.nn.pytorch import RelGraphConv
import pandas as pd
from model import BaseRGCN

import utils
max_nodes=180
# with open('3_neighbors.txt', 'rb') as handle:
#     neighbors = pickle.loads(handle.read())
    
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.h_dim=h_dim
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        #convkb
        self.out_channels=32
        self.conv= nn.Conv1d(4, self.out_channels, 1)  # kernel size x 3
        self.activation = nn.Tanh() 
        self.fc_layer = nn.Linear(h_dim * self.out_channels, 1, bias=False)
    def calc_score(self, g, h, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        # score = torch.sum(s * r * o, dim=1)
        #subgraph readout
        readout=self.subgraph_readout(g, h, embedding, triplets[:,0].squeeze(),triplets[:,2].squeeze())
        #convkb
        s=s.unsqueeze(1)
        r=r.unsqueeze(1)
        o=o.unsqueeze(1)
        readout=readout.unsqueeze(1)
        # readout=torch.zeros(s.shape).cuda()
        conv_input = torch.cat([s, r, o,readout], 1)
        out_conv= self.conv(conv_input)
        out_conv=self.activation(out_conv)
        in_fc=out_conv.view(-1,self.h_dim*self.out_channels)
        score = self.fc_layer(in_fc).view(-1)
        return -score

    def subgraph_readout(self, g, h, embedding, src, tgt):
        adj=g.adj(scipy_fmt="coo")
        l=adj.shape[0]
        adj=torch.sparse.LongTensor(torch.LongTensor((adj.row,adj.col)).cuda(),torch.tensor(adj.data).cuda(),adj.shape).to_dense().float()
        K=adj+torch.matrix_power(adj,2)+torch.matrix_power(adj,3)
        edges=torch.vstack((src,tgt)).T
        ind=(K[edges]>0).all(dim=1)
        indices_mask=torch.vstack([torch.arange(1,l+1)]*ind.shape[0]).cuda()
        ind=indices_mask*ind-1
        ind=torch.sort(ind,1,descending=True)[0]
        max_ind=max(torch.argmin(ind,dim=1))
        ind=ind[:,:max_ind]

        # batch_size=src.shape[0]
        # ind=torch.ones(batch_size,args.max_subgraph_size,dtype=int)
        mask=ind<0
        subgraph_emb=embedding[ind]
        subgraph_emb[mask,:]=0
        denominator=(subgraph_emb.sum(axis=2) > 0).sum(axis=1).view(-1,1).float()
        denominator[denominator==0]=1
        readout_emb=subgraph_emb.sum(axis=1) / denominator
        return readout_emb

    # def subgraph_readout(self, g, h, embedding, src, tgt):
    #     adj=g.adjacency_matrix(scipy_fmt='csr')
    #     batch=[]
    #     nodes=h.squeeze().cpu() #nodes in current minibatch graph
    #     print(nodes.shape)
    #     batch_size=src.shape[0]
    #     ind=-torch.ones(batch_size,args.max_subgraph_size,dtype=int)
    #     i=0
    #     # for s,t in zip(src,tgt):
    #     #     # nodes, _, _, _, _=utils.k_hop_subgraph(int(s),int(t),self.num_hops, adj)
    #     #     intersection=list(neighbors[int(s)]&neighbors[int(t)]&set(nodes)) #intersect k-hop nb of src and tgt
    #     #     subgraph=np.intersect1d(intersection,nodes) #nb of src and tgt in current minibatch graph
    #     #     local_nodes=np.searchsorted(nodes,subgraph) #convert to current node id in current minibatch
    #     #     ind[i,:len(local_nodes)]=local_nodes
    #     #     i+=1
    #     mask=ind<0
    #     # print(embedding)
    #     subgraph_emb=embedding[ind]
    #     subgraph_emb[mask,:]=0
    #     denominator=(subgraph_emb.sum(axis=2) > 0).sum(axis=1).view(-1,1).float()
    #     denominator[denominator==0]=1
    #     readout_emb=subgraph_emb.sum(axis=1) / denominator
    #     return readout_emb

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, h, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(g, h, embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    # load graph data
    if args.dataset is not None:
        data = load_data(args.dataset)
        num_nodes = data.num_nodes
        num_rels = data.num_rels
        train_data = data.train
        valid_data = data.valid
        test_data = data.test
    else:
        print("toy dataset")
        num_nodes = 6884#data.num_nodes
        num_rels = 990#data.num_rels
        train_data=pd.read_csv('../data/fb15k/toy/train.txt',sep='\t',names=['h','r','t'])
        train_data=train_data.to_numpy()
        valid_data=pd.read_csv('../data/fb15k/toy/valid.txt',sep='\t',names=['h','r','t'])
        valid_data=valid_data.to_numpy()
        test_data=pd.read_csv('../data/fb15k/toy/test.txt',sep='\t',names=['h','r','t'])
        test_data=test_data.to_numpy()
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create modelf
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    it=tqdm(range(args.n_epochs))
    # while True:
    for i in it:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        # print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        # with torch.autograd.detect_anomaly():
        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, node_id, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        it.set_description("Epoch {:04d} | Loss {:.4f} |  Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
            #   format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id, test_rel, test_norm)
            # mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
            #                      valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
            #                      eval_p=args.eval_protocol,model=model)
            # # save best model
            # if mrr < best_mrr:
            #     if epoch >= args.n_epochs:
            #         break
            # else:
            #     best_mrr = mrr
            #     torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
            #                model_state_file)
            utils.hitat10(test_graph, test_node_id, embed, model.w_relation, test_data, args.eval_batch_size, model=model)
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    # embed = model(test_graph, test_node_id, test_rel, test_norm)
    # utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
    #                test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=False,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=1,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--max-subgraph-size", type=int, default=200,
            help="number of hops in enclosing subgraph")

    args = parser.parse_args()
    print(args)
    main(args)