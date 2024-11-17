import torch
import pickle as pickle
import numpy as np

import constants

def load_data(batch_size, max_len):
    print("Loading train/test data...")

    thread_to_sub = {}

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
        
    # this is the subreddit's string ids

    with open(constants.SUBREDDIT_IDS) as fp:
        sub_id_map = {sub:i for i, sub in enumerate(fp.readline().split())}
    

    ## this is the user's string ids

    with open(constants.USER_IDS) as fp:
        user_id_map = {user:i for i, user in enumerate(fp.readline().split())}

    # source_sub, dest_sub, user, time, title, body
    with open(constants.PREPROCESSED_DATA) as fp:
        words, users, subreddits, lengths, labels, ids = [], [], [], [], [], []
        for i, line in enumerate(fp):
            info = line.split("\t")
            if info[1] in label_map and info[1] in source_to_dest_sub:
                title_words = info[-2].split(":")[1].strip().split(",")
                title_words = title_words[:min(len(title_words), max_len)]
                if len(title_words) == 0 or title_words[0] == '':
                    continue
                words.append(list(map(int, title_words)))

                body_words = info[-1].split(":")[1].strip().split(",")
                body_words = body_words[:min(len(body_words), max_len-len(title_words))]
                if not (len(body_words) == 0 or body_words[0] == ''):
                    words[-1].extend(list(map(int, body_words)))

                words[-1] = [constants.VOCAB_SIZE+1 if w==-1 else w for w in words[-1]]

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

                users.append([constants.NUM_USERS if not info[3] in user_id_map else user_id_map[info[3]]])
                ids.append(info[1])

                lengths.append(len(words[-1])+3)
                labels.append(label_map[info[1]])

        batches = []

        for count, i in enumerate(np.random.permutation(len(words))):
            if count % batch_size == 0:
                batch_words = np.ones((max_len, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_users = np.ones((1, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_subs = np.ones((2, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
                batch_lengths = []
                batch_labels = []
                batch_ids = []
            length = min(max_len, len(words[i]))
            batch_words[:length, count % batch_size] = words[i][:length]
            batch_users[:, count % batch_size] = users[i]
            batch_subs[:, count % batch_size] = subreddits[i]
            batch_lengths.append(length)
            batch_labels.append(labels[i])
            batch_ids.append(ids[i])
            if count % batch_size == batch_size - 1:
                order = np.flip(np.argsort(batch_lengths), axis=0)
                batches.append((list(np.array(batch_ids)[order]),
                    torch.LongTensor(batch_words[:,order]), 
                    torch.LongTensor(batch_users[:,order]), 
                    torch.LongTensor(batch_subs[:,order]), 
                    list(np.array(batch_lengths)[order]),
                    torch.FloatTensor(np.array(batch_labels)[order])))
    return batches