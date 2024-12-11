import torch

import constants

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