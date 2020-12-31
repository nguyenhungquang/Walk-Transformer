import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv, FastRGCNConv,GCNConv
from torch.functional import F
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train=pd.read_csv('data/fb15k/toy/train.txt',sep='\t',names=['h','r','t'])
train_edge_index=torch.tensor([np.array(train['h']),np.array(train['t'])]).to(device)
train_edge_type=torch.tensor(list(train['r'])).to(device)
test=pd.read_csv('data/fb15k/toy/test.txt',sep='\t',names=['h','r','t'])
test_edge_index=torch.tensor([np.array(test['h']),np.array(test['t'])]).to(device)
test_edge_type=torch.tensor(list(test['r'])).to(device)
e_size=6884
r_size=990
dim_size=128
batch_size=2048
lr=0.01
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb=nn.Embedding(e_size,dim_size)
        self.conv1 = RGCNConv(in_channels=dim_size, out_channels=dim_size,num_relations= r_size,num_bases=2)
        self.conv2 = RGCNConv(in_channels=dim_size, out_channels=dim_size,num_relations= r_size,num_blocks=4)
    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(self.emb(torch.arange(e_size).to(device)), edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return x
class Decoder(torch.nn.Module):
    def __init__(self,rel_size=r_size,feature_dim_size=dim_size):
        super(Decoder,self).__init__()
        self.rel_size=rel_size
        self.feature_dim_size=feature_dim_size
        self.rel=nn.Embedding(self.rel_size, self.feature_dim_size)
        self.criterion=nn.MarginRankingLoss(1.0)
    def forward(self, node_embs,edge_index,edge_type):
        pos_edge_type=edge_type
        pos_edge_index=edge_index
        pos_triplets=torch.cat((pos_edge_index[:1],pos_edge_type,pos_edge_index[1:]),dim=0)
        corrupted_list=(0,2)
        neg_triplets=self.sampling(pos_triplets,corrupted_list)
        neg_edge_index,neg_edge_type=neg_triplets[[0,2]],neg_triplets[1]
        pos=self.score(node_embs,pos_edge_type,pos_edge_index)
        neg=self.score(node_embs,neg_edge_type,neg_edge_index)
        # pos=torch.log(pos)
        # neg=torch.log(1-F.normalize(neg,dim=0))
        # return (torch.sum(pos)+torch.sum(neg))/((1+self.neg_num)*self.pos_num)
        score=self.criterion(pos,neg,torch.tensor([-1],device=device))
        return score
    def score(self, node_embs,edge_type,edge_index):
        src=torch.index_select(node_embs,0,edge_index[0])
        tgt=torch.index_select(node_embs,0,edge_index[1])
        rel=self.rel(edge_type).squeeze()
        out=src+rel-tgt
        return torch.norm(out,dim=1)
    def sampling(self,pos_triplets,corrupted_list):
        neg_triplets=pos_triplets.clone()
        batch_size=pos_triplets.shape[-1]
        split_size=batch_size//len(corrupted_list)
        for i,start in zip(corrupted_list,range(0,batch_size,split_size)):
            stop=min(start+split_size,batch_size)
            id_max=r_size-1 if i==1 else e_size-1
            neg_triplets[i,start:stop]=torch.randint(high=id_max,size=[stop-start],device=device)
            neg_triplets[i,start:stop]+=(neg_triplets[i,start:stop]>=pos_triplets[i,start:stop]).long()
        return neg_triplets
class Net(nn.Module):
    def __init__(self,rel_size=r_size,feature_dim_size=dim_size):
        super(Net,self).__init__()
        self.encoder=Encoder()
        # self.encoder=nn.Parameter(torch.rand(e_size,dim_size))
        self.decoder=Decoder(rel_size=r_size,feature_dim_size=dim_size)
    def forward(self,total_index,total_type,edge_index,edge_type):
        node_embs=self.encoder(total_index,total_type)
        # node_embs=self.encoder
        score=self.decoder(node_embs,edge_index,edge_type.unsqueeze(0))
        return score
    def hit_at_10(self,edge_index,edge_type,src,rel,tgt):
        node_embs=self.encoder(edge_index,edge_type)
        src=torch.index_select(node_embs,0,src)
        rel=self.decoder.rel(rel)
        # rel_embs=self.decoder.rel(torch.tensor([i for i in range(r_size)],device=device))
        # tgt=torch.index_select(node_embs,0,tgt)
        out=src+rel
        dist=torch.cdist(out,node_embs)
        id=torch.argsort(dist,axis=-1)[:,:10]
        cnt=torch.sum(torch.tensor([int(tgt[i]) in id[i] for i in range(len(tgt))],device=device))
        # print(cnt)
        return float(cnt/len(rel))
model=Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
def train():
    model.train()
    l=train_edge_index.shape[-1]
    perm=torch.randperm(l)
    total_loss=0
    for i in range(0,l,batch_size):
        optimizer.zero_grad()
        ind=perm[i:i+batch_size]
        batch_index=train_edge_index[:,ind]
        batch_type=train_edge_type[ind]
        out = model(train_edge_index,train_edge_type,batch_index,batch_type)
        loss = out
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/(l//batch_size+1)
def evaluate():
    model.eval()
    score=model.hit_at_10(train_edge_index,train_edge_type,test_edge_index[0],test_edge_type,test_edge_index[1])
    print(score)
    return score
for epoch in range(5001):
    print('Epoch: ',epoch)
    loss = train()
    print(loss)
    score=evaluate()