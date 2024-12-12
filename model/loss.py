import torch
import torch.nn as nn
import enum

class LossType(enum.Enum):
    SIMILARITY = 1
    SIMILARITY_WITH_NEGATIVE = 2
    SIMILARITY_WITH_NEGATIVE_USE_SIGMOID = 3

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, user_embeddings, subreddit_embeddings):
        similarity = self.cosine_similarity(user_embeddings, subreddit_embeddings)

        return 1 - similarity.mean()
    

class SimilarityLossWithNegative(nn.Module):
    def __init__(self, negative_sample=5):
        super(SimilarityLossWithNegative, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        self.negative_samples = negative_sample

    def forward(
            self,
            user_embeddings,
            subreddit_embeddings,
            batch_users,
            batch_subreddits,
            total_user_embeddings,
            total_subreddit_embeddings,
            graph,
        ):

        positive_similarity = self.cosine_similarity(user_embeddings, subreddit_embeddings)

        count = 0

        # Step 2: Generate negative samples based on the graph
        negative_loss = torch.tensor(0.0, requires_grad=True, device=user_embeddings.device)
        
        for user in batch_users:
            negative_user_embeddings = []
            connected_subreddits = graph.successors(user, etype='interacts')

            connected_subreddits_set = set(connected_subreddits.cpu().numpy())
            
            # Sample negative subreddits not connected to the current user
            all_subreddits = set(range(total_subreddit_embeddings.size(0)))

            negative_candidates = list(all_subreddits - connected_subreddits_set)
            
            negative_indices = torch.randperm(len(negative_candidates))[:self.negative_samples].clone().detach()

            sampled_negative_embeddings = total_subreddit_embeddings[negative_indices]

            negative_user_embeddings = total_user_embeddings[user].unsqueeze(0).expand_as(sampled_negative_embeddings)

            negative_loss = negative_loss + self.cosine_similarity(negative_user_embeddings, sampled_negative_embeddings).mean()

            count += 1
            if count % 1000 == 0:
                print("Processed {} users".format(count))

        positive_loss = 1 - positive_similarity.mean()
        negative_loss = negative_loss / count

        print(positive_loss, negative_loss)

        loss = positive_loss + negative_loss

        return loss
    
class SimilarityLossUseSigmoid(nn.Module):
    def __init__(self, negative_sample=5):
        super(SimilarityLossUseSigmoid, self).__init__()
        self.negative_samples = negative_sample

    def forward(
            self,
            user_embeddings,
            subreddit_embeddings,
            batch_users,
            batch_subreddits,
            total_user_embeddings,
            total_subreddit_embeddings,
            graph,
        ):

        constant = 1e-8

        # Compute positive similarity using element-wise product and sigmoid
        positive_product = (user_embeddings * subreddit_embeddings).sum(dim=-1)
        positive_similarity = torch.sigmoid(positive_product)  # Apply sigmoid
        positive_log_similarity_loss = - torch.log(positive_similarity + constant)

        count = 0

        # Step 2: Generate negative samples based on the graph
        negative_loss = torch.tensor(0.0, requires_grad=True, device=user_embeddings.device)
        
        for user in batch_users:
            connected_subreddits = graph.successors(user, etype='interacts')
            connected_subreddits_set = set(connected_subreddits.cpu().numpy())

            # Sample negative subreddits not connected to the current user
            all_subreddits = set(range(total_subreddit_embeddings.size(0)))
            negative_candidates = list(all_subreddits - connected_subreddits_set)
            negative_indices = torch.randperm(len(negative_candidates))[:self.negative_samples].clone().detach()

            sampled_negative_embeddings = total_subreddit_embeddings[negative_indices]
            negative_user_embeddings = total_user_embeddings[user].unsqueeze(0).expand_as(sampled_negative_embeddings)

            # Compute negative similarity using element-wise product and sigmoid
            negative_product = (negative_user_embeddings * sampled_negative_embeddings).sum(dim=-1)
            negative_similarity = torch.sigmoid(- negative_product) 
            negative_log_similarity_loss = torch.log(negative_similarity + constant)
            negative_loss = negative_loss + negative_log_similarity_loss.mean()

            count += 1

            if count % 1000 == 0:
                print("Processed {} users".format(count))

        # Compute final positive and negative losses
        positive_loss = positive_log_similarity_loss.mean()
        negative_loss = negative_loss / count

        print(positive_loss, negative_loss)

        # Combine positive and negative losses
        loss = positive_loss + negative_loss

        return loss