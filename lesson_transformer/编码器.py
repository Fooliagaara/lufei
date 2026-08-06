from torch import nn
import copy
from lesson_transformer.规范化层 import LayerNorm


def clones(moudel, N):
    return nn.ModuleList([copy.deepcopy(moudel) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)