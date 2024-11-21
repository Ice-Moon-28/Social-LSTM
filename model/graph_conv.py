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