import numpy as np
import torch
import torch.nn as nn

import constants
from model.embedding import Embeddings
from util.load_self_trained_embedding import format_embedding_file_name, load_embeddings

class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_dims, output_dim, activation=nn.ReLU, dropout=0.0, prepend_social=False, args=None):
        """
        :param input_dim: 输入的维度
        :param hidden_dims: 隐藏层维度列表，例如 [128, 64, 32]
        :param output_dim: 输出的维度
        :param activation: 激活函数，默认为 ReLU
        :param dropout: Dropout 概率，默认为 0.0
        """
        super(MultiLayerPerceptron, self).__init__()
        
        layers = []

        self.args = args

        self.prepend_social = prepend_social

        if self.prepend_social:
            prev_dim = constants.WORD_EMBED_DIM * (3)
        else:
            prev_dim = constants.WORD_EMBED_DIM * 0

        glove_embeds = self._load_glove_embeddings()
        self.glove_embeds= torch.FloatTensor(glove_embeds)
        self.pad_embed = torch.zeros(1, constants.WORD_EMBED_DIM)
        self.unk_embed = torch.FloatTensor(1,constants.WORD_EMBED_DIM)
        self.unk_embed.normal_(std=1./np.sqrt(constants.WORD_EMBED_DIM))
        self.word_embeds = nn.Parameter(torch.cat([self.glove_embeds, self.pad_embed, self.unk_embed], dim=0), requires_grad=False)
        self.embed_module = torch.nn.Embedding(constants.VOCAB_SIZE+2, constants.WORD_EMBED_DIM)
        self.embed_module.weight = self.word_embeds

        user_embeds = self._load_user_embeddings()
        self.user_embeds = torch.nn.Embedding(constants.NUM_USERS+1, constants.WORD_EMBED_DIM)
        self.user_embeds.weight  = nn.Parameter(torch.cat([torch.FloatTensor(user_embeds),  
            self.pad_embed]), requires_grad=False)

        subreddit_embeds = self._load_subreddit_embeddings()
        self.subreddit_embeds = torch.nn.Embedding(constants.NUM_SUBREDDITS+1, constants.WORD_EMBED_DIM)
        self.subreddit_embeds.weight  = nn.Parameter(torch.cat([torch.FloatTensor(subreddit_embeds), 
            self.pad_embed]), requires_grad=False)


        # 添加隐藏层
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(prev_dim, hidden_dim)

            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

            layers.append(linear_layer) 

            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _generate_position_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension
        return pe
    
    def _load_glove_embeddings(self):
        print("Loading word embeddings...")
        with open(constants.WORD_EMBEDS) as fp:
            embeddings = np.empty((constants.VOCAB_SIZE, constants.WORD_EMBED_DIM), dtype=np.float32)
            for i, line in enumerate(fp):
                embeddings[i,:] = list(map(float, line.split()[1:]))
        return embeddings


    def _load_user_embeddings(self):
        print("Loading user embeddings...")

        if self.args.embedding_self_trained:
            embeds = load_embeddings(
                filename=format_embedding_file_name(
                    embedding_type=self.args.embedding_type,
                    negative_sample=self.args.negative_sample,
                    loss_type=self.args.loss_type,
                    hidden_feats=self.args.hidden_feats
                )
            )

            return embeds['user_embeddings']
        else:
            embeds = Embeddings(constants.USER_EMBEDS)
            return embeds._vecs

    def _load_subreddit_embeddings(self):
        print("Loading subreddit embeddings...")

        if self.args.embedding_self_trained:
            embeds = load_embeddings(
                filename=format_embedding_file_name(
                    embedding_type=self.args.embedding_type,
                    negative_sample=self.args.negative_sample,
                    hidden_feats=self.args.hidden_feats,
                    loss_type=self.args.loss_type,
                )
            )

            return embeds['subreddit_embeddings']
        else:
            embeds = Embeddings(constants.SUBREDDIT_EMBEDS)
            return embeds._vecs
        
    def forward(self, text_inputs, user_inputs, subreddit_inputs, metafeats, lengths):
        
        batch_size = text_inputs.shape[1]

        # text_inputs = self.embed_module(text_inputs).permute(1, 0, 2).reshape(batch_size, -1)
        user_inputs = self.user_embeds(user_inputs).permute(1, 0, 2).reshape(batch_size, -1)
        subreddit_inputs = self.subreddit_embeds(subreddit_inputs).permute(1, 0, 2).reshape(batch_size, -1)

        if self.prepend_social:
            inputs = torch.cat([user_inputs, subreddit_inputs], dim=1)
        else:
            inputs = text_inputs

        return self.model(inputs)

