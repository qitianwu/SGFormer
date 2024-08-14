import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from performer_pytorch import SelfAttention

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, in_channels,
                 num_heads,
                 dropout=0.0,
                 attn_dropout=0.0, use_bn=True):
        super().__init__()

        self.dim_h = in_channels
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.batch_norm = use_bn

        # Local message-passing model.
        self.local_model=GCNConv(in_channels,in_channels)

        # Global attention transformer-style model.
        self.self_attn = SelfAttention(
            dim=in_channels, heads=num_heads,
            dropout=self.attn_dropout, causal=False)

        # Normalization for MPNN and Self-Attention representations.
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(in_channels)
            self.norm1_attn = nn.BatchNorm1d(in_channels)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(in_channels, in_channels * 2)
        self.ff_linear2 = nn.Linear(in_channels * 2, in_channels)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(in_channels)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        self.device=None

    def reset_parameters(self):
        for child in self.children():
            # print(child.__class__.__name__)
            classname=child.__class__.__name__
            if classname not in ['SelfAttention','Dropout']:
                child.reset_parameters()
        
        if self.device is None:
            param=next(iter(self.local_model.parameters()))
            self.device=param.device

        self.self_attn=SelfAttention(
            dim=self.dim_h, heads=self.num_heads,
            dropout=self.attn_dropout, causal=False).to(self.device)

    def forward(self, x, edge_index):
        h_in1 = x  # for first residual connection, x has shape (n, in_channels)

        h_out_list = []
        # Local MPNN with edge attributes.
        h_local=self.local_model(x,edge_index)
        h_local=h_in1+h_local # Residual connection.

        if self.batch_norm:
            h_local=self.norm1_local(h_local)
        h_out_list.append(h_local)

        h_attn=self.self_attn(x.unsqueeze(0)) # (1, n, in_channels)
        h_attn=h_attn.squeeze(0) # (n, in_channels)

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.batch_norm:
            h = self.norm2(h)

        return h

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
class GPSModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, num_heads,dropout,attn_dropout,use_bn):
        super().__init__()

        self.pre_mp=nn.Linear(in_channels,hidden_channels)
        self.dropout=dropout
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GPSLayer(
                hidden_channels,
                num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                use_bn=use_bn,
            ))

        self.post_mp = nn.Linear(hidden_channels,out_channels)

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index=data.graph['edge_index']
        x=self.pre_mp(x)
        x=F.relu(x)
        x=F.dropout(x,self.dropout,training=self.training)
        for layer in self.layers:
            x=layer(x,edge_index)
        x=self.post_mp(x)
        return x
    
    def reset_parameters(self):
        self.pre_mp.reset_parameters()
        self.post_mp.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
