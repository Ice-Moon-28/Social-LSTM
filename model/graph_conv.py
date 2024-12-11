import torch.nn as nn
import torch
import dgl

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        # 为每个边类型创建卷积层
        self.conv1 = dgl.nn.HeteroGraphConv({
            ('user', 'interacts', 'subreddit'): dgl.nn.GraphConv(in_feats, hidden_feats),
            ('subreddit', 'interacted_by', 'user'): dgl.nn.GraphConv(in_feats, hidden_feats)
        })
        self.conv2 = dgl.nn.HeteroGraphConv({
            ('user', 'interacts', 'subreddit'): dgl.nn.GraphConv(hidden_feats, out_feats),
            ('subreddit', 'interacted_by', 'user'): dgl.nn.GraphConv(hidden_feats, out_feats)
        })

    def forward(self, g, features):
        h = self.conv1(g, features) 
        h = {k: torch.relu(v) for k, v in h.items()}  
        h = self.conv2(g, h)
        return h
    

import torch.nn as nn
import torch
import dgl

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads):
        super(GAT, self).__init__()
        # 为每种边类型定义 GAT 层
        self.gat1 = dgl.nn.HeteroGraphConv({
            ('user', 'interacts', 'subreddit'): dgl.nn.GATConv(in_feats, hidden_feats, num_heads),
            ('subreddit', 'interacted_by', 'user'): dgl.nn.GATConv(in_feats, hidden_feats, num_heads)
        })
        self.gat2 = dgl.nn.HeteroGraphConv({
            ('user', 'interacts', 'subreddit'): dgl.nn.GATConv(hidden_feats * num_heads, out_feats, num_heads),
            ('subreddit', 'interacted_by', 'user'): dgl.nn.GATConv(hidden_feats * num_heads, out_feats, num_heads)
        })
        self.num_heads = num_heads

def forward(self, g, features):
    # 应用第一个 GAT 层并激活
    h = self.gat1(g, features)
    h = {k: torch.relu(v.view(v.size(0), -1)) for k, v in h.items()}  # 合并多头输出

    # 应用第二个 GAT 层
    h = self.gat2(g, h)
    h = {k: v.mean(dim=1) for k, v in h.items()}  # 多头求平均作为输出

    return h