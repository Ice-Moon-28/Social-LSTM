import torch
import random
import argparse
import pickle as pickle
import torch.nn as nn
import numpy as np

def set_seed(seed=42):
    random.seed(seed)                      # 设置 Python 随机数种子
    np.random.seed(seed)                   # 设置 NumPy 随机数种子
    torch.manual_seed(seed)                # 设置 PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)           # 设置 PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)       # 设置所有 GPU 随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 的卷积操作确定性
    torch.backends.cudnn.benchmark = False
