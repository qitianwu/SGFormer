'''
Merge GCN and Transformer as GraphTrans
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TransFormer(nn.Module):
    def __init__(self,input_channels,d_model,out_channels,num_layers,n_head,dim_feedforward,dropout,activation='relu',norm_input=True) -> None:
        super().__init__()
        self.gnn2transformer=nn.Linear(input_channels,d_model)
        self.trans_layer=nn.TransformerEncoderLayer(d_model,n_head,dim_feedforward,dropout,activation)
        self.layer_norm=nn.LayerNorm(d_model)
        self.transformer=nn.TransformerEncoder(self.trans_layer,num_layers,self.layer_norm)
        if norm_input:
            self.input_layernorm=nn.LayerNorm(d_model)
        self.norm_input=norm_input
        self.output=nn.Linear(d_model,out_channels)
    
    def reset_parameters(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self,x):
        '''
        x: (seq, batch, feature)
        '''
        x=self.gnn2transformer(x)
        if self.norm_input:
            x=self.input_layernorm(x)
        x=self.transformer(x) #(seq,batch,d_model)
        x_out=self.output(x)
        return x_out
    
    

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=True, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.convs.append(
            #     GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # self.convs.append(
        #     GCNConv(hidden_channels, out_channels, cached=not save_mem, normalize=not save_mem))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index=data.graph['edge_index']
        edge_weight=data.graph['edge_weight'] if 'edge_weight' in data.graph else None
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is None:
                x = conv(x, edge_index)
            else:
                x=conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x

class graphTrans(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_emb_dim=64, d_model=64, num_layers=2, 
                 num_trans_layers=3, num_trans_head=4, dim_feedforward=256, dropout=0.5, trans_dropout=0.1, use_bn=True):
        super().__init__()
        self.trans=TransFormer(gnn_emb_dim,d_model,out_channels,num_trans_layers,
                               num_trans_head,dim_feedforward,trans_dropout)
        self.gnn=GCN(in_channels,hidden_channels,gnn_emb_dim,num_layers,dropout,use_bn=use_bn)

    def forward(self, data):
        node_emb=self.gnn(data)
        node_emb=node_emb.unsqueeze(1)
        out=self.trans(node_emb)
        out=out.squeeze(1)

        return out
    
    def reset_parameters(self):
        self.trans.reset_parameters()
        self.gnn.reset_parameters()