import torch
import torch.nn as nn
import math



# 定义Embeddings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层, 他们共享参数.
# 该类继承nn.Module, 这样就有标准层的一些功能, 这里我们也可以理解为一种模式, 我们自己实现的所有层都会这样去写.
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        """类的初始化函数, 有两个参数, d_model: 指词嵌入的维度, vocab: 指词表的大小."""
        # 接着就是使用super的方式指明继承nn.Module的初始化函数, 我们自己实现的所有层都会这样去写.
        super(Embeddings, self).__init__()
        # 之后就是调用nn中的预定义层Embedding, 获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
           当传给该类的实例化对象参数时, 自动调用该类函数
           参数x: 因为Embedding层是首层, 所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回

        # 让 embeddings vector 在增加 之后的 postion encoding 之前相对大一些的操作，
        # 主要是为了让position encoding 相对的小，这样会让原来的 embedding vector 中的信息在和 position encoding 的信息相加时不至于丢失掉
        # 让 embeddings vector 相对大一些
        return self.lut(x) * math.sqrt(self.d_model)


if __name__ == '__main__':
    embedding1 = nn.Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    out = embedding1(input)
    print(out)
    print("*" * 20)
    embedding2 = Embeddings(10, 10)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    out = embedding2(input)
    print(out)