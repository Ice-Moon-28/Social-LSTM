import torch

import constants

def format_embedding_file_name(embedding_type, negative_sample, loss_type, hidden_feats):
    return "self_trained_embeddings/" + f"embedding_type_{embedding_type}_negative_sample_{negative_sample}_loss_{loss_type}_hidden_feats_{hidden_feats}.pt"


def load_embeddings(filename='embeddings.pt'):
    """
    Load embeddings from the specified file.

    :param filename: Path to the saved embeddings file.
    :return: A dictionary with user and subreddit embeddings and IDs.
    """
    try:
        # 使用 torch.load 直接加载 .pt 文件

        states = torch.load(filename, map_location="cpu")

        # 提取保存的内容
        user_embeddings = states['user_embeddings']
        user_ids = states['user_ids']
        subreddit_embeddings = states['subreddit_embeddings']
        subreddit_ids = states['subreddit_ids']

        return {
            'user_embeddings': user_embeddings[:constants.NUM_USERS,:],
            'user_ids': user_ids,
            'subreddit_embeddings': subreddit_embeddings[:constants.NUM_SUBREDDITS,:],
            'subreddit_ids': subreddit_ids,
        }
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except KeyError as e:
        print(f"Error: Missing key {e} in the saved embeddings file.")
        return None
    except RuntimeError as e:
        print(f"Error loading the file: {e}")
        return None
    
if __name__ == "__main__":
    print(format_embedding_file_name(2, 5, 128))