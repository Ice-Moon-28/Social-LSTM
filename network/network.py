import constants
import dgl
import torch
import torch.nn as nn
import numpy as np
from model.embedding import Embeddings
from model.embedding import EmbeddingType
from model.graph_conv import GCN
from model.loss import LossType, SimilarityLoss, SimilarityLossUseSigmoid, SimilarityLossWithNegative
from util.load_self_trained_embedding import format_embedding_file_name
from util.sparse_matrix import sparse_eye

device = constants.embedding_device

class RedditNetwork:
    def __init__(
            self,
            epochs=10,
            batch_size=64,
            learning_rate=0.01,
            loss_type=LossType.SIMILARITY,
            embedding_type=EmbeddingType.RANDOM_INITIALIZE,
            negative_samples=5,
            hidden_feats=128
        ):
        user_source_sub = self.load_graph_data()

        print("Building graph... on", device)

        self.graph = self.build_graph(user_source_sub).to(device)

        self.num_users = self.graph.num_nodes('user')
        self.num_subreddits = self.graph.num_nodes('subreddit')

        self.features = self.load_embeddings(embedding_type=embedding_type)

        self.model = GCN(in_feats=self.features['user'].shape[1], hidden_feats=hidden_feats, out_feats=300).to(device) 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.batch_size = batch_size

        self.num_epochs = epochs

        self.loss_fn = self.load_loss_fn(loss_type=loss_type, negative_samples=negative_samples)


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
        source_sub, user = zip(*edges)

        graph = dgl.heterograph({
            ('user', 'interacts', 'subreddit'): (user, source_sub),
            ('subreddit', 'interacted_by', 'user'): (source_sub, user),
        })

        return graph

    def train(self):  
        self.model.train()
        
        # 获取所有边
        edges = self.graph.edges(etype='interacts')

        # edges = self.mask_invalid_edge(edges)

        edges = (edges[0].to(device), edges[1].to(device))
        total_edges = len(edges[0])

        best_loss = np.Infinity

        for epoch in range(self.num_epochs):
            total_loss = 0.0

            batch_count = 0
            
            perm = torch.randperm(total_edges)
            
            for i in range(0, total_edges, self.batch_size):
                batch_indices = perm[i:i+self.batch_size]

                batch_count += 1
                
                batch_users = edges[0][batch_indices]
                batch_subreddits = edges[1][batch_indices]

                self.optimizer.zero_grad()

                node_embeddings = self.model(self.graph, self.features)
                
                user_embeddings = node_embeddings['user'][batch_users].to(device)
                subreddit_embeddings = node_embeddings['subreddit'][batch_subreddits].to(device)

                batch_loss = self.loss_fn(
                    user_embeddings,
                    subreddit_embeddings,
                    batch_users=batch_users,
                    batch_subreddits=batch_subreddits,
                    total_user_embeddings=node_embeddings['user'],
                    total_subreddit_embeddings=node_embeddings['subreddit'],
                    graph=self.graph,
                )

                batch_loss.backward()
                
                self.optimizer.step()

                total_loss += batch_loss.item()

                print(f"Batch {batch_count} loss:", batch_loss.item())

                print(f"Total loss:{epoch+1}", total_loss / batch_count)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {total_loss / batch_count:.4f}")

            eval_loss = self.eval_embedding()

            if eval_loss < best_loss:
                best_loss = eval_loss
                
                self.save_model()
                self.save_embeddings()

                print("New Best loss and Saved model and embeddings", "Best loss:", best_loss.item())

        return eval_loss
    
    def eval_embedding(self):
        self.model.eval()
        with torch.no_grad():
            edges = self.graph.edges(etype='interacts')
            
            # edges = self.mask_invalid_edge(edges)
            
            batch_users = edges[0]
            batch_subreddits = edges[1]

            node_embeddings = self.model(self.graph, self.features)

            user_embeddings = node_embeddings['user'][batch_users].to(device)
            subreddit_embeddings = node_embeddings['subreddit'][batch_subreddits].to(device)

            batch_loss = self.loss_fn(
                user_embeddings,
                subreddit_embeddings,
                batch_users=batch_users,
                batch_subreddits=batch_subreddits,
                total_user_embeddings=node_embeddings['user'],
                total_subreddit_embeddings=node_embeddings['subreddit'],
                graph=self.graph,
            )

            print("Eval loss:", batch_loss.item())

        return batch_loss

    
    def save_model(self, filename='model.pt'):
        torch.save(self.model.state_dict(), filename)
    
    def save_embeddings(self, filename='embeddings.pt'):
        self.model.eval() 
        with torch.no_grad():
            node_embeddings = self.model(self.graph, self.features)

        user_embeddings = node_embeddings['user']
        subreddit_embeddings = node_embeddings['subreddit']
        
        user_ids = self.graph.nodes('user')
        subreddit_ids = self.graph.nodes('subreddit') 

        states = {
            'user_embeddings': user_embeddings,
            'user_ids': user_ids,
            'subreddit_embeddings': subreddit_embeddings,
            'subreddit_ids': subreddit_ids,
        }

        torch.save(
            states,
            filename=format_embedding_file_name(
                embedding_type=self.embedding_type,
                negative_sample=self.negative_sample,
                hidden_feats=self.hidden_feats,
                loss_type=self.loss_type,
            )
        )
    
    def load_embeddings(self, embedding_type=EmbeddingType.RANDOM_INITIALIZE):
        self.pad_embeds = torch.zeros(1, constants.WORD_EMBED_DIM)

        if embedding_type == EmbeddingType.RANDOM_INITIALIZE:
            self.user_features = torch.randn(self.num_users, 300, requires_grad=True).to(device)  # 用户特征
            self.subreddit_features = torch.randn(self.num_subreddits, 300, requires_grad=True).to(device) # 社区特征

            self.user_features = torch.cat([self.user_features, self.pad_embeds], dim=0)
            self.subreddit_features = torch.cat([self.subreddit_features, self.pad_embeds], dim=0)

            import pdb; pdb.set_trace()

            return {
                'user': self.user_features,
                'subreddit': self.subreddit_features
            }
        
        elif embedding_type == EmbeddingType.PRETRAINED:
            self.user_features = Embeddings(constants.USER_EMBEDS)._vecs
               
            self.subreddit_features = Embeddings(constants.SUBREDDIT_EMBEDS)._vecs

            self.user_features = torch.tensor(self.user_features, dtype=torch.float32).to(device=device)
            self.subreddit_features = torch.tensor(self.subreddit_features, dtype=torch.float32).to(device=device)
            self.pad_embeds = self.pad_embeds.to(device=device)

            self.user_features = torch.cat([self.user_features, self.pad_embeds], dim=0)
            self.subreddit_features = torch.cat([self.subreddit_features, self.pad_embeds], dim=0)

            return {
                'user': self.user_features,
                'subreddit': self.subreddit_features
            }
        
        elif embedding_type == EmbeddingType.ONE_HOT:
            self.num_users = self.graph.num_nodes('user')
            self.num_subreddits = self.graph.num_nodes('subreddit')

            self.user_features = torch.tensor([[i] for i in range(self.num_users)], dtype=torch.float32).to(device=device)
            self.subreddit_features = torch.tensor([[i] for i in range(self.num_subreddits)], dtype=torch.float32).to(device=device)

            return {
                'user': self.user_features,
                'subreddit': self.subreddit_features
            }
        
        elif embedding_type == EmbeddingType.ONE_HOT_WITH_BIGGER_SIZE:
            self.num_users = self.graph.num_nodes('user')
            self.num_subreddits = self.graph.num_nodes('subreddit')

            dimension = max(self.num_users, self.num_subreddits)

            self.user_features = sparse_eye(self.num_users, dimension).to(device=device)
            self.subreddit_features = sparse_eye(self.num_subreddits, dimension).to(device=device)

            return {
                'user': self.user_features,
                'subreddit': self.subreddit_features
            }

    def load_loss_fn(self, loss_type=LossType.SIMILARITY, negative_samples=5):
        if loss_type == LossType.SIMILARITY:
            return SimilarityLoss()
        elif loss_type == LossType.SIMILARITY_WITH_NEGATIVE:
            return SimilarityLossWithNegative(negative_sample=negative_samples)
        elif loss_type == LossType.SIMILARITY_WITH_NEGATIVE_USE_SIGMOID:
            return SimilarityLossUseSigmoid(negative_sample=negative_samples)
        
    def mask_invalid_edge(self, edges):
        edges_src, edges_des = edges

        mask = (edges_src != constants.NUM_USERS) & (edges_des != constants.NUM_SUBREDDITS)

        return (edges_src[mask], edges_des[mask])