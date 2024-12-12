import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import constants
from model.embedding import Embeddings
from util.get_mlp import get_mlp
from util.load_self_trained_embedding import format_embedding_file_name, load_embeddings

class SocialTransformer(nn.Module):
    def __init__(
            self,
            hidden_dim=constants.WORD_EMBED_DIM,
            nhead=8,
            num_layers=2, 
            ## 是否在训练的入参中增加 social embedding
            prepend_social=True, 
            final_dense=True,
            ## 是否在线性层中增加 附带上 embeddings
            feedward_hidden_dim=1024,
            args=None,
            dropout_rate=0.1,
            include_text=True,
            linear_include_embeds=False,
            linear_layers_number=2,
        ):
        """
        Transformer model for predicting conflict between Reddit communities.
        Can incorporate social embeddings of users and communities/subreddits.

        hidden_dim - size of transformer layers
        nhead - number of heads in the multiheadattention models
        num_layers - number of transformer layers
        batch_size - size of minibatches during training
        prepend_social - if True then user/subreddit embeds are prepended.
        include_meta - if True then metadata/linguistic/hand-engineered features are included
        final_dense - whether to include an extra dense Linear+ReLU layer before the softmax
        include_embeds - whether to include the user/subreddit layers in the final (i.e, post-transformer) layer(s)
        """
        super(SocialTransformer, self).__init__()
        
        # Load embeddings
        self.args = args

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
        
        self.hidden_dim = hidden_dim
        self.prepend_social = prepend_social
        self.linear_include_embeds = linear_include_embeds
        self.final_dense = final_dense

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=constants.WORD_EMBED_DIM, nhead=nhead, dim_feedforward=feedward_hidden_dim, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer configuration
        input_dim = 1200 if self.linear_include_embeds else constants.WORD_EMBED_DIM

        self.linear_layers = get_mlp(
            input_dim=input_dim,
            hidden_dims=[512 * 64, 512 ],
            output_dim=constants.NUM_CLASSES,
            dropout=dropout_rate
        )
        

        self.dropout = nn.Dropout(dropout_rate) 

        self.include_text = include_text

    def _load_glove_embeddings(self):
        print("Loading word embeddings...")
        with open(constants.WORD_EMBEDS) as fp:
            embeddings = np.empty((constants.VOCAB_SIZE, constants.WORD_EMBED_DIM), dtype=np.float32)
            for i, line in enumerate(fp):
                embeddings[i,:] = list(map(float, line.split()[1:]))
        return embeddings
    
    def _generate_position_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension
        return pe


    def _load_user_embeddings(self):
        print("Loading user embeddings...")

        if self.args.embedding_self_trained:
            embeds = load_embeddings(
                filename=format_embedding_file_name(
                    embedding_type=self.args.embedding_type,
                    negative_sample=self.args.negative_sample,
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
                    hidden_feats=self.args.hidden_feats
                )
            )

            return embeds['subreddit_embeddings']
        else:
            embeds = Embeddings(constants.SUBREDDIT_EMBEDS)
            return embeds._vecs

    def forward(self, text_inputs, user_inputs, subreddit_inputs, metafeats, lengths):
        # Embedding lookup
        # self.debug(user_inputs, subreddit_inputs)

        inputs_list = []

        text_inputs = self.embed_module(text_inputs)
        user_inputs = self.user_embeds(user_inputs)
        subreddit_inputs = self.subreddit_embeds(subreddit_inputs)

        if self.prepend_social:
            inputs_list.append(user_inputs)
            inputs_list.append(subreddit_inputs)

        if self.include_text:
            inputs_list.append(text_inputs)

        inputs = torch.cat(inputs_list, dim=0)

        sequence_length = inputs.size(0)

        self.pos_encoder = self._generate_position_encoding(sequence_length, constants.WORD_EMBED_DIM)

        self.pos_encoder = self.pos_encoder.permute(1, 0, 2)[:inputs.size(0), :, :].expand(-1, inputs.size(1), -1)

        self.pos_encoder = self.pos_encoder.to(constants.device)

        inputs = inputs + self.pos_encoder

        # Pass through Transformer Encoder
        encoded_output = self.transformer_encoder(inputs)

        if self.args.enable_mean_pooling:
            encoded_output = encoded_output.mean(dim=0)  # Pooling over sequence length
        else:
            encoded_output = encoded_output[0]
        # Concatenate meta features if needed
        final_input = encoded_output

        if self.linear_include_embeds:
            final_input = torch.cat([final_input, user_inputs.squeeze(), subreddit_inputs[0], subreddit_inputs[1]], dim=1)
        
        # final_input = self.dropout(final_input)

        weights = self.linear_layers(final_input)
        
        return weights
