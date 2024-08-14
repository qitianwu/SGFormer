import math
import os
from multiprocessing.sharedctypes import Value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GCN
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

class GATAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(GATAttention, self).__init__()
        self.num_heads = num_heads
        self.dk = in_channels // num_heads
        self.out_channels = out_channels

        # Learnable parameters
        self.Wq = nn.Linear(in_channels, num_heads * self.dk)
        self.Wk = nn.Linear(in_channels, num_heads * self.dk)
        self.Wv = nn.Linear(in_channels, num_heads * out_channels)
        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self):
        self.Wq.reset_parameters()
        self.Wk.reset_parameters()
        self.Wv.reset_parameters()

    def forward(self, qs, ks, vs):
        # Linear projections
        qs = self.Wq(qs).view(-1, self.num_heads, self.dk)
        ks = self.Wk(ks).view(-1, self.num_heads, self.dk)
        vs = self.Wv(vs).view(-1, self.num_heads, self.out_channels)

        # Scaled dot-product attention
        attn_scores = torch.einsum("nhm,lhm->nlh", qs, ks) / torch.sqrt(nn.Parameter(torch.Tensor([self.dk])).to(qs.device))
        attn_weights = self.softmax(attn_scores)

        # Attentive aggregation
        output = torch.einsum("nlh,lhd->nhd", attn_weights, vs)
        return output

class TransConvLayerGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super(TransConvLayerGAT, self).__init__()
        self.attention = GATAttention(in_channels, out_channels, num_heads)
        self.use_weight = use_weight

        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.attention.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input):
        # Feature transformation
        query = query_input
        key = source_input
        if self.use_weight:
            value = self.Wv(source_input)
        else:
            value = source_input

        # Attention
        attention_output = self.attention(query, key, value)
        return attention_output

class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super(TransConvLayer, self).__init__()
        self.attention = TransConvLayerGAT(in_channels, out_channels, num_heads, use_weight)
        self.use_weight = use_weight
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

    def reset_parameters(self):
        self.attention.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # Feature transformation
        query = query_input
        key = source_input
        if self.use_weight:
            value = self.Wv(source_input)
        else:
            value = source_input

        # Attention
        if output_attn:
            attention_output, attn = self.attention(query, key)
        else:
            attention_output = self.attention(query, key)

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act=use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        edge_weight = data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class SGFormerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, 
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv=TransConv(in_channels,hidden_channels,num_layers,num_heads,alpha,dropout,use_bn,use_residual,use_weight)
        self.gnn=gnn
        self.use_graph=use_graph
        self.graph_weight=graph_weight
        self.use_act=use_act

        self.aggregate=aggregate

        if aggregate=='add':
            self.fc=nn.Linear(hidden_channels,out_channels)
        elif aggregate=='cat':
            self.fc=nn.Linear(2*hidden_channels,out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')
        
        self.params1=list(self.trans_conv.parameters())
        self.params2=list(self.gnn.parameters()) if self.gnn is not None else []
        self.params2.extend(list(self.fc.parameters()) )

    def forward(self,data):
        x1=self.trans_conv(data)
        if self.use_graph:
            x2=self.gnn(data)
            if self.aggregate=='add':
                x=self.graph_weight*x2+(1-self.graph_weight)*x1
            else:
                x=torch.cat((x1,x2),dim=1)
        else:
            x=x1
        x=self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns=self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()