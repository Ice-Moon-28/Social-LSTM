import torch


def sparse_eye(n, m):
    # 生成主对角线的索引 (i, i)
    indices = torch.arange(min(n, m))  # 对角线索引
    values = torch.ones(min(n, m))  # 对角线上的值为1
    
    # 构造索引（二维：行和列位置）
    row_indices = indices  # 行索引
    col_indices = indices  # 列索引
    
    # 组合行列索引
    indices = torch.stack([row_indices, col_indices])  # (2, N)
    
    # 创建稀疏矩阵
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (n, m))
    return sparse_matrix