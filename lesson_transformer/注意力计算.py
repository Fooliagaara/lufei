import torch.nn.functional as F
import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    # query的最后一维的大小，一般情况下就等同于我们的词嵌入维度，命名为d_k
    d_k = query.size(-1)

    # 将k转置后才能进行两个矩阵相乘，即Q * K
    scores = torch.metmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn
