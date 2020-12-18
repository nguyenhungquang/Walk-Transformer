import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from log_uniform import LogUniformSampler

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

"""LogUniformSampler is taken from https://github.com/rdspring1/PyTorch_GBW_LM"""


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nrels, nsampled, nhid, device):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.device = device
        #
        self.sampler = LogUniformSampler(self.ntokens)
        #
        self.weight = nn.Parameter(torch.Tensor(ntokens, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        # sample ids according to word distribution - Unique
        sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
        return self.sampled(inputs, labels, sample_values)

    """@Dai Quoc Nguyen: Implement the sampled softmax loss function as described in the paper
    On Using Very Large Target Vocabulary for Neural Machine Translation https://www.aclweb.org/anthology/P15-1001/"""

    def sampled(self, inputs, labels, sample_values):
        assert inputs.data.get_device() == labels.data.get_device()

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values
        sample_ids = Variable(torch.LongTensor(sample_ids)).to(self.device)
        # gather true labels
        true_weights = torch.index_select(self.weight, 0, labels)
        # true_weights=F.normalize(true_weights,dim=-1)
        # gather sample ids
        sample_weights = torch.index_select(self.weight, 0, sample_ids)
        # sample_weights = F.normalize(sample_weights,dim=-1)

        #original
        # true_dots = torch.sum(torch.mul(inputs, true_weights), dim=1, keepdim=True)
        # sample_dots = torch.matmul(inputs, torch.t(sample_weights))
        # row_max_vals = torch.max(
        #     torch.cat([true_dots, sample_dots], dim=1), dim=1, keepdim=True
        # )[0]

        # numerator = true_dots - row_max_vals
        # denominator = torch.logsumexp(sample_dots - row_max_vals, dim=1, keepdim=True)
        # log_llh = numerator - denominator

        # return -log_llh

        #transe loss
        true_logits = torch.exp(torch.norm(inputs-true_weights,dim=1))
        inputs=inputs.unsqueeze(1)#view(-1,1,128)
        sample_weights=sample_weights.unsqueeze(0)#view(1,-1,128)
        sample_logits=torch.sum(torch.exp(torch.norm(inputs-sample_weights,dim=-1)).squeeze(),dim=1)

        logits = torch.log(true_logits / sample_logits)
        # print(true_logits.shape,sample_logits.shape,logits.shape)
        return logits
