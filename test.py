data_path = "F:/实习/practice/toutiao/data/data/"
train_path = data_path + "train.txt"  # 训练集
dev_path = data_path + "dev.txt"  # 验证集
test_path = data_path + "test.txt"  # 测试集
class_list = [
    x.strip() for x in open(data_path + "class.txt").readlines()
]  # 类别名单


print("ni hao")