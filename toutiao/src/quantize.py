# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, required=True, help="choose a model: bert")
args = parser.parse_args()


if __name__ == "__main__":
    dataset = "toutiao"  # 数据集
    if args.model == "bert":
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        # 数据迭代器的预处理和生成
        print("Loading data for Bert Model...")
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        # 实例化模型并加载参数, 注意不要加载到GPU之上, 只能在CPU上实现模型量化
        model = x.Model(config)
        model.load_state_dict(torch.load(config.save_path))

        # 量化BERT模型
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

        print(quantized_model)

        # 测试量化后的模型在测试集上的表现
        test(config, quantized_model, test_iter)
        # 保存量化后的模型
        torch.save(quantized_model, config.save_path_quantization)