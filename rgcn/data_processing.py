from utils import *
from torch.utils.data import Dataset, DataLoader
class KGData(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index]
class Batch:
    def __init__(self, num_rels, adj_list, degrees, split_size, negative_rate):
        self.num_rels=num_rels
        self.adj_list=adj_list
        self.degrees=degrees
        self.split_size=split_size
        self.negative_rate=negative_rate
        # self.device=device
    def __subgraph_index(self, g, data):
        data=torch.from_numpy(data).T.cuda()
        src=data[0]
        tgt=data[2]
        adj=g.adj(scipy_fmt="coo")
        l=adj.shape[0]
        adj=torch.sparse.LongTensor(torch.LongTensor((adj.row,adj.col)).cuda(),torch.tensor(adj.data).cuda(),adj.shape).to_dense().float()
        K=torch.matrix_power(adj+torch.eye(adj.shape[0],device=torch.device("cuda")),3)
        edges=torch.stack((src,tgt)).T
        ind=(K[edges]>0).any(dim=1) #all: intersection, any: union
        indices_mask=torch.stack([torch.arange(1,l+1)]*ind.shape[0]).cuda()
        # print(ind.shape,indices_mask.shape)
        ind=indices_mask*ind-1
        ind=torch.sort(ind,1,descending=True)[0]
        ind=ind[:,~(ind==-1).all(dim=0)]
        return ind
    def __call__(self, data):
        edges=np.stack(data)#.to(self.device)
        length=edges.shape[0]
        src, rel, dst = edges.transpose()
        # print(src,dst)
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            self.negative_rate)

        # further split graph, only half of the edges will be used as graph
        # structure, while the rest half is used as unseen positive samples
        split_size = int(length * self.split_size)
        graph_split_ids = np.random.choice(np.arange(length),
                                          size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

        # build DGL graph
        # print("# sampled nodes: {}".format(len(uniq_v)))
        # print("# sampled edges: {}".format(len(src) * 2))
        g, rel, norm = build_graph_from_triplets(len(uniq_v), self.num_rels,
                                                (src, rel, dst))
        # subgraph_id= self.__subgraph_index(g, samples)
        return g, uniq_v, rel, norm, samples, labels