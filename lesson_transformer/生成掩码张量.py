import numpy as np
import torch


def subsequent_mask(size):
    # 首先定义掩码张量的形状
    attn_shape = (1, size, size)
    # 使用np.ones方法向其中添加1元素，形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(1 - subsequent_mask)

     