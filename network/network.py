import constants
import dgl
import torch
import torch.nn as nn
from model.graph_conv import GCN
from model.loss import SimilarityLoss
import torch.autograd as autograd
autograd.set_detect_anomaly(True)

class RedditNetwork:
    def __init__(self, epochs=10, batch_size=64, learning_rate=0.01):
        user_source_sub = self.load_graph_data()

        self.graph = self.build_graph(user_source_sub)
        self.num_users = self.graph.num_nodes('user')
        self.num_subreddits = self.graph.num_nodes('subreddit')

        self.model = GCN(in_feats=300, hidden_feats=128, out_feats=300) 

        self.user_features = torch.randn(self.num_users, 300, requires_grad=True)  # 用户特征
        self.subreddit_features = torch.randn(self.num_subreddits, 300, requires_grad=True)  # 社区特征

        self.features = {
            'user': self.user_features,
            'subreddit': self.subreddit_features
        }

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.batch_size = batch_size

        self.num_epochs = epochs

        self.loss_fn = SimilarityLoss()


    def load_graph_data(self, max_len=512):
        print("Loading graph data...")

        thread_to_sub = {}
        user_source_sub = []
        # source_sub, dest_sub, user, time, title, body
        
        # this is cross_link in the post
        with open(constants.POST_INFO) as fp:
            for line in fp:
                info = line.split()
                source_sub = info[0]
                target_sub = info[1]
                source_post = info[2].split("T")[0].strip()
                target_post = info[6].split("T")[0].strip()
                thread_to_sub[source_post] = source_sub
                thread_to_sub[target_post] = target_sub
        
        label_map = {}
        source_to_dest_sub = {}

        with open(constants.SUBREDDIT_IDS) as fp:
            sub_id_map = {sub:i for i, sub in enumerate(fp.readline().split())}

        ## this is the user's string ids

        with open(constants.USER_IDS) as fp:
            user_id_map = {user:i for i, user in enumerate(fp.readline().split())}

        # this is the crosslink's attribution
        with open(constants.LABEL_INFO) as fp:
            for line in fp:
                info = line.split("\t")
                source = info[0].split(",")[0].split("\'")[1]
                dest = info[0].split(",")[1].split("\'")[1]
                label_map[source] = 1 if info[1].strip() == "burst" else 0
                try:
                    source_to_dest_sub[source] = thread_to_sub[dest]
                except KeyError:
                    continue
        
        with open(constants.PREPROCESSED_DATA) as fp:
            words, users, subreddits, lengths, labels, ids = [], [], [], [], [], []
            for i, line in enumerate(fp):
                info = line.split("\t")
                if info[1] in label_map and info[1] in source_to_dest_sub:
                    if not info[0] in sub_id_map:
                        source_sub = constants.NUM_SUBREDDITS
                    else:
                        source_sub = sub_id_map[info[0]]
                    dest_sub = source_to_dest_sub[info[1]]
                    if not dest_sub in sub_id_map:
                        dest_sub = constants.NUM_SUBREDDITS
                    else:
                        dest_sub = sub_id_map[dest_sub]
                    subreddits.append([source_sub, dest_sub])

                    user = constants.NUM_USERS if not info[2] in user_id_map else user_id_map[info[2]]

                    user_source_sub.append((source_sub, user))

            
        return user_source_sub

    def build_graph(self, edges):
        # 分离边数据：source为社区，dest为用户
        source_sub, user = zip(*edges)

        # 创建二部图 (社区节点 -> 用户节点)
        graph = dgl.heterograph({
            ('user', 'interacts', 'subreddit'): (user, source_sub),
            ('subreddit', 'interacted_by', 'user'): (source_sub, user),
        })

        return graph

    def train(self):  # 将批次大小调小
        self.model.train()
        
        # 获取所有边
        edges = self.graph.edges(etype='interacts')
        total_edges = len(edges[0])

        batch_count = 0

        best_loss = float('inf')


        for epoch in range(self.num_epochs):
            total_loss = 0.0
            
            perm = torch.randperm(total_edges)
            
            # 分批次处理
            for i in range(0, total_edges, self.batch_size):
                batch_indices = perm[i:i+self.batch_size]

                batch_count += 1
                
                batch_users = edges[0][batch_indices]
                batch_subreddits = edges[1][batch_indices]

                self.optimizer.zero_grad()

                node_embeddings = self.model(self.graph, self.features)
                
                user_embeddings = node_embeddings['user'][batch_users]
                subreddit_embeddings = node_embeddings['subreddit'][batch_subreddits]

                batch_loss = self.loss_fn(user_embeddings, subreddit_embeddings)

                batch_loss.backward()
                
                self.optimizer.step()

                total_loss += batch_loss.item()

                print("Batch loss:", batch_loss.item())

                print("Total loss:", total_loss / batch_count)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {total_loss / (total_edges/self.batch_size):.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                
                torch.save(self.model.state_dict(), f'{best_loss}/best_model.pt')

        return total_loss