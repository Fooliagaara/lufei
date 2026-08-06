import torch
import torch.nn as nn
import torch.nn.functional as F


# 通过类PositionwiseFeedForward来实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输⼊参数分别是d_model, d_ff,和dropout=0.1，第⼀个是线性层的输⼊维
       度也是第⼆个线性层的输出维度，
        因为我们希望输⼊通过前馈全连接层后输⼊和输出的维度不变. 第⼆个参数d_ff就是第⼆个线性
       层的输⼊维度和第⼀个线性层的输出维度.
        最后⼀个是dropout置0⽐率."""

        super(PositionwiseFeedForward, self).__init__()

        # ⾸先按照我们预期使⽤nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使⽤nn的Dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输⼊参数为x，代表来⾃上⼀层的输出"""
        # ⾸先经过第⼀个线性层，然后使⽤Funtional中relu函数进⾏激活,
        # 之后再使⽤dropout进⾏随机置0，最后通过第⼆个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))