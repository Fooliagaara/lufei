# coding: UTF-8
import torch
import torch.nn as nn
import os
from transformers import T5EncoderModel, BertTokenizer, T5Config


class Config(object):
    def __init__(self, dataset):
        self.model_name = "t5"
        self.data_path = "../data/data/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [
            x.strip() for x in open(self.data_path + "class.txt").readlines()
        ]  # 类别名单
        self.save_path = "../saved_dict"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path_quantization = self.save_path + "/" + self.model_name + "_quantized.pt"
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
        # 模型量化的时候, 放开下一行代码, 在CPU上运行.
        # self.device = 'cpu'

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5  # 学习率
        self.dropout = 0.1  # dropout概率
        self.t5_path = "../data/t5_chinese_pretrain"
        # uer/t5-base-chinese-cluecorpussmall 用的是 BERT 词表(vocab.txt), tokenizer_class=BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.t5_path)
        self.t5_config = T5Config.from_pretrained(self.t5_path)
        self.hidden_size = self.t5_config.d_model


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.t5 = T5EncoderModel.from_pretrained(config.t5_path, config=config.t5_config)
        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 输入的句子
        context = x[0]
        # 对padding部分进行mask, 和句子一个size, padding部分用0表示, 比如[1, 1, 1, 1, 0, 0]
        mask = x[2]

        # T5是encoder-decoder结构, 这里只取encoder, 输出last_hidden_state, 没有pooler_output
        t5_out = self.t5(context, attention_mask=mask).last_hidden_state  # [batch, seq_len, d_model]

        # 平均池化: 对非padding位置的向量求均值, 得到句向量 [batch, d_model]
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (t5_out * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        out = self.fc(pooled)
        return out
