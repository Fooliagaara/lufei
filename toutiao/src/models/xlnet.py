# coding: UTF-8
import torch
import torch.nn as nn
import os
from transformers import XLNetTokenizer,XLNetModel,XLNetConfig



class Config(object):
    def __init__(self, dataset):
        self.model_name = "xlnet"
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
        self.xlnet_path = "../data/xlnet_chinese_large"

        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.xlnet_config = XLNetConfig.from_pretrained(self.xlnet_path)
        self.hidden_size = self.xlnet_config.d_model


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path, config=config.xlnet_config)
        self.dropout = nn.Dropout(config.dropout)

        # 查看xlnet内部参数
        # for name, param in self.xlnet.named_parameters():
        #     print(name)

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x[0]: input_ids [batch, seq_len]
        # x[2]: padding mask (attention_mask, 1=有效token，0=pad)
        input_ids = x[0]
        attention_mask = x[2]

        # XLNet 前向: 没有pooler_output, 输出last_hidden_state
        xlnet_out = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = xlnet_out.last_hidden_state  # [batch, seq_len, d_model]

        # 平均池化: 对非padding位置的向量求均值, 得到句向量 [batch, d_model]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        out = self.fc(pooled)
        return out

