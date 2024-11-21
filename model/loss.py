import torch.nn as nn

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, user_embeddings, subreddit_embeddings):
        similarity = self.cosine_similarity(user_embeddings, subreddit_embeddings)

        return 1 - similarity.mean()