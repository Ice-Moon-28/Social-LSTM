import torch
import random
import argparse
import pickle as pickle
import torch.nn as nn
import numpy as np
from transformers import get_scheduler

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from constants import device

import constants
from model.gpt import all_positive, prompt_gpt, random_guess
from model.model_choices import ModelChoices
from embeddings import Embeddings
from model.social_lstm import SocialLSTM
from util.load_data import load_data
from model.transformer import SocialTransformer
from torch.optim.lr_scheduler import LambdaLR

import random
import numpy as np
import torch

from util.summary_writer import log_learning_rate, log_scale_test, log_scale_train

from transformers import get_scheduler





def set_seed(seed=42):
    random.seed(seed)                      # 设置 Python 随机数种子
    np.random.seed(seed)                   # 设置 NumPy 随机数种子
    torch.manual_seed(seed)                # 设置 PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)           # 设置 PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)       # 设置所有 GPU 随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 的卷积操作确定性
    torch.backends.cudnn.benchmark = False



def get_embeddings(data):
    embeds = []
    ids = []
    for batch in data:
        id, text, users, subs, lengths, metafeats, labels = batch
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
        model(text, users, subs, metafeats, lengths)
        batch_embeds = model.h
        embeds.append(batch_embeds.t().data.cpu().numpy())
        ids.extend(id)
    return ids, np.concatenate(embeds)

def train(model, train_data, val_data, test_data, optimizer,
        epochs=10, log_every=100, log_file=None, save_embeds=False):
    if not log_file is None:
        lg_str = log_file
        log_file = open(log_file, "w")

    ema_loss = None
    if constants.ENABLE_CROSS_ENTROPY:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    best_iter = (0., 0,0)
    best_test = 0.
    embeds = None
    for epoch in range(epochs):
        random.shuffle(train_data)
        for i, batch in enumerate(train_data):
            model.train()

            _, text, users, subs, lengths, metafeats, labels = batch
            text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
            optimizer.zero_grad()
            outputs = model(text, users, subs, metafeats, lengths)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            # scheduler.step()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.01*loss.data.item() + 0.99*ema_loss

            if i % 10 == 0:
                print(epoch, i, ema_loss)
                print(epoch, i, ema_loss, file=log_file)
            if  i % log_every == 0:
                auc = evaluate_auc(model, val_data)

                log_scale_train(epoch, loss.data.item(), auc)

                log_learning_rate(optimizer, epoch=epoch)

                print("Val AUC", epoch, i, auc)
                if not log_file is None:
                    print("Val AUC", epoch, i, auc, file=log_file)
                if auc > best_iter[0]:
                    best_iter = (auc, epoch, i)
                    print("New best val!", best_iter)
                    best_test = evaluate_auc(model, test_data)
                    # if auc > 0.7:
                    #     ids, embeds = get_embeddings(train_data+val_data+test_data)
    print("Overall best val:", best_iter)
    if not log_file is None:
        print("Overall best test:", best_test, file=log_file)
        print("Overall best val:", best_iter, file=log_file)
        if not embeds is None and save_embeds:
            np.save(open(lg_str+"-embeds.npy", "w"), embeds)
            pickle.dump(ids, open(lg_str+"-ids.pkl", "w"))
    return best_iter[0]

def evaluate_auc(model, test_data):

    model.eval()

    predictions = []
    gold_labels = []
    for batch in test_data:
        _, text, users, subs, lengths, metafeats, labels = batch
        if (constants.CUDA and device.type == "cuda") or device.type == "mps":
            gold_labels.extend(labels.cpu().numpy().tolist())
        else:
            gold_labels.extend(labels.numpy().tolist())
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
        
        if constants.ENABLE_CROSS_ENTROPY:
            outputs = model(text, users, subs, metafeats, lengths)[:,1]
        else:
            outputs = model(text, users, subs, metafeats, lengths)

        if (constants.CUDA and device.type == "cuda") or device.type == "mps":
            predictions.extend(outputs.data.squeeze().cpu().numpy().tolist())
        else:
            predictions.extend(outputs.data.squeeze().numpy().tolist())

    auc = roc_auc_score(gold_labels, predictions)
    return auc

if __name__ == "__main__":
    set_seed(seed=constants.SEED)


    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--log_file", type=str, default=None, 
            help="Where to log the model training details.")
    parser.add_argument("--model", type=int, default=ModelChoices.Transformer,
            help="choose the model")
    parser.add_argument("--save_embeds", action='store_true',
            help="Whether to save the hidden-state LSTM embeddings that are generated.\
                  They will be stored based on the log_file name used above.")
    parser.add_argument("--dropout", type=float, default=0.2,
            help="Dropout rate for inter-LSTM layers in 2-layer LSTM.")
    parser.add_argument("--single_layer", action='store_true',
            help="Use single-layer LSTM (implies that dropout param is ignored)")
    parser.add_argument("--include_meta", action='store_true',
            help="Include metadata/hand-crafted features in final layer of model.")
    parser.add_argument("--final_dense", action='store_true',
            help="Include an extra Linear+ReLU layer before the softmax.")
    parser.add_argument("--lstm_append_social", action='store_true', 
            help="Append the social embeddings instead of prepending them to LSTM input.")
    parser.add_argument("--lstm_no_social", action='store_true', 
            help="Do not include social embeddings in LSTM input.")
    parser.add_argument("--final_layer_social", action='store_true', 
            help="(Also) include social embeddings in the final layer.")
    
    parser.add_argument("--enable_mean_pooling", type=bool, default=False,
            help="Enable transformer encoder's mean pooling")
    
    parser.add_argument("--total_steps", type=int, default=0, help='total steps')
    parser.add_argument("--warmup_steps", type=int, default=1000, help='warmup steps')

    parser.add_argument("--dropout_rate", type=float, default=0.1,
            help="The dropout_rate of the transformer model")
   
    args = parser.parse_args()
    dropout = None if args.single_layer else args.dropout
    if args.lstm_append_social and args.lstm_no_social:
        raise Exception("Only one of --lstm_append_social and --lstm_no_social can be True at a time.")
    if args.log_file is None and args.save_embeds:
        raise Exception("A log file must be specified if you want to store the LSTM embeddings of the posts.")
    if args.lstm_append_social or args.lstm_no_social:
        prepend_social = None if args.lstm_no_social else False
    else:
        prepend_social = True


    print("Loading training data")
    # WE HAVE PRE-CONSTRUCTED TRAIN/VAL/TEST DATA USING load_data
    # this avoids re-doing all the pre-processing everytime the code is
    # run. This data is fixed to a batch size of 512.
    train_data = pickle.load(open(constants.TRAIN_DATA, 'rb'))
    val_data = pickle.load(open(constants.VAL_DATA, 'rb'))
    test_data = pickle.load(open(constants.TEST_DATA, 'rb'))

    print(len(train_data)*constants.BATCH_SIZE, "training examples", len(val_data)*512, "validation examples")
    print(sum([i for batch in train_data for i in batch[-1]]), "positive training", sum([i for batch in val_data for i in batch[-1]]), "positive validation")

    # annoying checks for CUDA switches....
    for i in range(len(train_data)):
        batch = train_data[i]
        metafeats = batch[5]
        train_data[i] = (batch[0], 
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device),
                batch[4],
                metafeats.to(device),
                batch[6].to(device))

    for i in range(len(val_data)):
        batch = val_data[i]
        metafeats = batch[5]
        val_data[i] = (batch[0], 
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device),
                batch[4],
                metafeats.to(device),
                batch[6].to(device))

    for i in range(len(test_data)):
        batch = test_data[i]
        metafeats = batch[5]
        test_data[i] = (batch[0], 
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device),
                batch[4],
                metafeats.to(device),
                batch[6].to(device))

    best_auc = (0,"") 

    print(f"Loading model into {device}")

    
    if (ModelChoices(args.model) == ModelChoices.LSTM):
        model = SocialLSTM(args.hidden_dim, prepend_social=prepend_social, dropout=args.dropout, include_embeds=args.final_layer_social, 
            include_meta=args.include_meta, final_dense=args.final_dense)
        
        model.to(device)
        
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate)

        # def lr_lambda(current_step: int):
        #     warmup_steps = args.warmup_steps
        #     if current_step < warmup_steps:
        #         return current_step / warmup_steps  # 学习率逐步增加
        #     return 1.0  # 保持初始学习率

        # # 使用 LambdaLR 设置学习率调度器
        # scheduler = LambdaLR(optimizer, lr_lambda)
      
        auc = train(model, train_data, val_data, test_data, optimizer, epochs=args.epochs, log_file=args.log_file, save_embeds=args.save_embeds,
                # scheduler=scheduler,
        )

    elif (ModelChoices(args.model) == ModelChoices.Transformer):
        model = SocialTransformer(
            hidden_dim=300,
            feedward_hidden_dim=1024,
            nhead=6,
            num_layers=1,
            prepend_social=prepend_social,
            include_meta=args.include_meta,
            final_dense=args.final_dense,
            include_embeds=args.final_layer_social,
            args=args,
            dropout_rate=args.dropout
        )

        model.to(device)

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.learning_rate)

        # def lr_lambda(current_step: int):
        #     warmup_steps = args.warmup_steps
        #     if current_step < warmup_steps:
        #         return current_step / warmup_steps  # 学习率逐步增加
        #     return 1.0  # 保持初始学习率

        # # 使用 LambdaLR 设置学习率调度器
        # scheduler = LambdaLR(optimizer, lr_lambda)
    
        auc = train(model, train_data, val_data, test_data, optimizer, epochs=args.epochs, log_file=args.log_file, save_embeds=args.save_embeds,
            # scheduler=scheduler,
        )
    elif (ModelChoices(args.model) == ModelChoices.GPT_4o) or (ModelChoices(args.model) == ModelChoices.GPT_4o_mini):
        prompt_gpt(choice=ModelChoices(args.model))
    elif (ModelChoices(args.model) == ModelChoices.Random):
        random_guess()
    elif (ModelChoices(args.model) == ModelChoices.AllPositive):
        all_positive()

    



    
