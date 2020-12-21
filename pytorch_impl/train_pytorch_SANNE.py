#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)

from tqdm import tqdm, trange
import numpy as np

np.random.seed(123)
import time
import pickle as cPickle
from pytorch_model_SANNE import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
import statistics
from generate_random_walks import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("SANNE")
parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="cora", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=3, type=int, help="Batch Size")
parser.add_argument(
    "--num_epochs", default=50, type=int, help="Number of training epochs"
)
parser.add_argument("--model_name", default="cora", help="")
parser.add_argument("--sampled_num", default=512, type=int, help="")
parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--num_heads", default=2, type=int, help="")
parser.add_argument(
    "--num_self_att_layers", default=1, type=int, help="Number of self-attention layers"
)
parser.add_argument(
    "--ff_hidden_size",
    default=16,
    type=int,
    help="The hidden size for the feedforward layer",
)
parser.add_argument("--num_neighbors", default=4, type=int, help="")
parser.add_argument("--fold_idx", type=int, default=1, help="The fold index. 0-9.")
parser.add_argument("--num_walks", type=int, default=3, help="")
parser.add_argument("--walk_length", type=int, default=8, help="")
args = parser.parse_args()
directory = "toy"
if args.dataset == 'fb':
    directory = "numerical/"
print(args)

# walks = generate_random_walks(input='../data/'+args.dataset+'.Full.edgelist', num_walks=args.num_walks, walk_length=args.walk_length)
# walks = generate_random_walks(input='../data/fb15k/freebase_mtr100_mte100-train.txt', num_walks=args.num_walks, walk_length=args.walk_length,kg=True)
walks = generate_random_walks(
    input="../data/fb15k/"+directory+"train.txt",
    num_walks=args.num_walks,
    walk_length=args.walk_length,
    kg=True,
)
data_size = np.shape(walks)[0]
# print(data_size)
# print(walks.shape)
# print(walks)

# cora,citeseer,pubmed
# with open('../data/'+args.dataset+'.128d.feature.pickle', 'rb') as f:
#     features_matrix = torch.from_numpy(cPickle.load(f)).to(device)
# vocab_size = features_matrix.size(0)
# feature_dim_size = features_matrix.size(1)
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


vocab_size = 14951  # file_len('../data/fb15k/entity2id.txt')
rel_size = 1345  # file_len('../data/fb15k/relation2id.txt')


class Batch_Loader_RW(object):
    def __init__(self):

        self.dict_neighbors = {}
        with open("../data/" + args.dataset + ".Full.edgelist", "r") as f:
            for line in f:
                lst_nodes = line.strip().split()
                if len(lst_nodes) == 2:
                    if int(lst_nodes[0]) not in self.dict_neighbors:
                        self.dict_neighbors[int(lst_nodes[0])] = []
                    self.dict_neighbors[int(lst_nodes[0])].append(int(lst_nodes[1]))

    def __call__(self):
        idxs = np.random.permutation(data_size)[: args.batch_size]
        context_nodes = []
        for walk in walks[idxs]:
            for node in walk:
                context_nodes.append(
                    np.random.choice(
                        self.dict_neighbors[node], args.num_neighbors, replace=True
                    )
                )
        return (
            torch.from_numpy(walks[idxs]).to(device),
            torch.from_numpy(np.array(context_nodes)).view(-1).to(device),
        )


class Batch_KB(object):
    def __init__(self):
        self.dict_neighbors = {}
        with open("../data/fb15k/"+directory+"train.txt", "r") as f:
            for line in f:
                trip = line.strip().split()
                if len(trip) == 3:
                    if int(trip[0]) not in self.dict_neighbors:
                        self.dict_neighbors[int(trip[0])] = []
                    if int(trip[2]) not in self.dict_neighbors:
                        self.dict_neighbors[int(trip[2])] = []
                    self.dict_neighbors[int(trip[0])].append(
                        [int(trip[1]), int(trip[2])]
                    )
                    self.dict_neighbors[int(trip[2])].append(
                        [int(trip[1]), int(trip[0])]
                    )

    def __call__(self):
        idxs = np.random.permutation(data_size)[: args.batch_size]
        context_nodes = []
        context_rels = []
        for walk in walks[idxs]:
            for node in walk:
                neighbor_nodes = np.array([i[1] for i in self.dict_neighbors[node]])
                neighbor_rels = np.array([i[0] for i in self.dict_neighbors[node]])
                neighbor_index = np.random.choice(
                    range(len(self.dict_neighbors[node])),
                    args.num_neighbors,
                    replace=True,
                )
                context_nodes.append(neighbor_nodes[neighbor_index])
                context_rels.append(neighbor_rels[neighbor_index])
        return (
            torch.from_numpy(walks[idxs]).to(device),
            torch.from_numpy(np.array(context_rels)).view(-1).to(device),
            torch.from_numpy(np.array(context_nodes)).view(-1).to(device),
        )


# batch_loader = Batch_Loader_RW()
batch_loader = Batch_KB()
# x,r,y=batch_loader()
# print(x,r,y)
print("Loading data... finished!")
model = SANNE(
    feature_dim_size=128,
    ff_hidden_size=args.ff_hidden_size,
    num_heads=args.num_heads,
    dropout=args.dropout,
    num_self_att_layers=args.num_self_att_layers,
    vocab_size=vocab_size,
    rel_size=rel_size,
    sampled_num=args.sampled_num,  # initialization=features_matrix,
    num_neighbors=args.num_neighbors,
    device=device,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # Adagrad?
num_batches_per_epoch = int((data_size - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=num_batches_per_epoch, gamma=0.1
)
from torch import autograd


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.0
    iterator = trange(num_batches_per_epoch)
    for _ in iterator:
        input_x, input_r, input_y = batch_loader()
        optimizer.zero_grad()
        logits = model(input_x, input_r, input_y)
        loss = torch.sum(logits)
        if math.isnan(loss):
            print(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        # print(loss.item())
        iterator.set_description("Training... (loss=%2.5f)" % loss.item())
        # break
    return total_loss


def evaluate(epoch, acc_write):
    model.eval()  # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.ss.weight
        node_embeddings = node_embeddings.data.cpu().numpy()
        idxs_10_data_splits = open("../data/" + args.dataset + ".10sampledtimes", "rb")
        # The evaluation process for each data split is to take the best score on the validation set from all hyper-parameter settings (to give the final score on the test set).
        for fold_idx in range(10):
            (
                train_idxs,
                train_labels,
                val_idxs,
                val_labels,
                test_idxs,
                test_labels,
            ) = cPickle.load(idxs_10_data_splits)

            train_embs = node_embeddings[list(train_idxs)]
            val_embs = node_embeddings[list(val_idxs)]
            test_embs = node_embeddings[list(test_idxs)]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_embs, train_labels)
            val_acc = cls.score(val_embs, val_labels)
            test_acc = cls.score(test_embs, test_labels)
            print(
                "epoch ",
                epoch,
                " fold_idx ",
                fold_idx,
                " val_acc ",
                val_acc * 100.0,
                " test_acc ",
                test_acc * 100.0,
            )
            acc_write.write(
                "epoch "
                + str(epoch)
                + " fold_idx "
                + str(fold_idx)
                + " val_acc "
                + str(val_acc * 100.0)
                + " test_acc "
                + str(test_acc * 100.0)
                + "\n"
            )

    return acc_write


def eval_KB():
    model.eval()
    # test_data=pd.read_csv('../data/fb15k/test',sep='\t',names=['s','r','t'])
    # correct_test=0
    # for index, row in test_data.iterrows():
    #     correct_test+=model.hit_at_10(row['s'],row['r'],row['t'])
    train_data = pd.read_csv("../data/fb15k/"+directory+'test.txt', sep="\t", names=["s", "r", "t"])
    correct_train = 0
    progress=tqdm(
        enumerate(train_data.itertuples(index=False)), desc="Evaluating...", total=len(train_data)
    )
    for i,row in progress:
        correct_train += model.hit_at_10(*row, device=device)
        progress.set_description("Hit@10: %2.5f"%(correct_train/(i+1)))
    print(correct_train / len(train_data))
    # print(correct_test/len(test_data))
# model.load_state_dict(torch.load('model.pt'))
# eval_KB()
# assert False

"""main process"""
import os

out_dir = os.path.abspath(
    os.path.join(args.run_folder, "../runs_pytorch_SANNE", args.model_name)
)
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
acc_write = open(checkpoint_prefix + "_acc.txt", "w")

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    print(train_loss)
    # acc_write = evaluate(epoch, acc_write)
    
    if epoch ==1:
        torch.save(model.state_dict(), 'model.pt')
    eval_KB()
    if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
        scheduler.step()

acc_write.close()