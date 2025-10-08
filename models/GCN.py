import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, output_dim, adj,bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
        torch.nn.init.kaiming_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.adj=adj

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(self.adj, support.transpose(0, 1)).t()
        if self.bias is not None:
            return output + self.bias
        else:
            # return output
            return output+input #简单残差
