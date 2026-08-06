import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from lesson_transformer.多头注意力 import MultiHeadedAttention
from lesson_transformer.前馈全连接层 import PositionwiseFeedForward
from lesson_transformer.编码器层 import EncoderLayer
from lesson_transformer.编码器 import Encoder
from lesson_transformer.解码器层 import DecoderLayer
from lesson_transformer.解码器 import Decoder
from lesson_transformer.线性层和softmax层 import Generator



# 使用EncoderDecoder类来实现编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """初始化函数中有5个参数, 分别是编码器对象, 解码器对象,
           源数据嵌入函数, 目标数据嵌入函数,  以及输出部分的类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """在forward函数中，有四个参数, source代表源数据, target代表目标数据,
           source_mask和target_mask代表对应的掩码张量"""

        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
size = 512
d_model = 512
d_ff = 64
dropout = 0.1
head = 8
c = copy.deepcopy
# 编码器构造
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
el = EncoderLayer(size, c(attn), c(ff), dropout)  # 由于每一层的注意力和前馈层内的参数都是不同的，所以应该每一层都再拷贝一次
N = 8
en = Encoder(el, N)
encoder = en
# 解码器构造
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask
dl = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
de = Decoder(dl, N)
decoder = de
# 编码器和解码器的文本嵌入层构造，使用的是nn中自带的，不是自己写的文本嵌入
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
# 输出层构造
d_model = 512
vocab_size = 1000
gen = Generator(vocab_size, d_model)
generator = gen

# 假设源数据与目标数据相同, 实际中并不相同
source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

# 假设src_mask与tgt_mask相同，实际中并不相同
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

# 编码器-解码器层的构造
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)