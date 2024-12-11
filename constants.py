import torch

# CONSTANTS YOU NEED TO MODIFY

#whether to train on GPU
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("mps")

CUDA = True

embedding_device = torch.device("cuda")
# root directory that contains the training/testing data
DATA_HOME="data/prediction/"
LOG_DIR="./logs"
#whether to show results on the test set
PRINT_TEST=False

# CONSTANTS YOU MAY WANT TO MODIFY (BUT DON"T NEED TO)
TRAIN_DATA=DATA_HOME+"/preprocessed_train_data.pkl"
VAL_DATA=DATA_HOME+"/preprocessed_val_data.pkl"
TEST_DATA=DATA_HOME+"/preprocessed_test_data.pkl"
BATCH_SIZE=512
SEED=3407
#NOTE: THESE PREPROCESSED FILES HAVE A FIXED BATCH SIZE

WORD_EMBEDS=DATA_HOME+"/embeddings/glove_word_embeds.txt"

USER_EMBEDS=DATA_HOME+"/embeddings/user_vecs.npy"
USER_IDS=DATA_HOME+"/embeddings/user_vecs.vocab"

SUBREDDIT_EMBEDS=DATA_HOME+"/embeddings/sub_vecs.npy"
SUBREDDIT_IDS=DATA_HOME+"/embeddings/sub_vecs.vocab"

POST_INFO=DATA_HOME+"/detailed_data/post_crosslinks_info.tsv"
LABEL_INFO=DATA_HOME+"/detailed_data/label_info.tsv"
PREPROCESSED_DATA=DATA_HOME+"/detailed_data/tokenized_posts.tsv"

VOCAB_SIZE = 174558
NUM_USERS = 118381
NUM_SUBREDDITS = 51278
WORD_EMBED_DIM = 300
METAFEAT_LEN = 263
ENABLE_CROSS_ENTROPY = True
NUM_CLASSES = 2 if ENABLE_CROSS_ENTROPY else 1
MAX_LEN=50


