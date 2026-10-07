import os

# 设置项⽬代码库的root路径, 为后续所有的包导⼊提供便利
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)

# 设置原始数据⽂件的路径, 通过以项⽬root路径为基础, 逐级添加到⽂件路径
train_raw_data_path = os.path.join(root_path, "data", "train.txt")
test_raw_data_path = os.path.join(root_path, "data", "test.txt")

# 停⽤词路径和jieba分词⽤户⾃定义字典路径
stop_words_path = os.path.join(root_path, "data", "stopwords.txt")
user_dict_path = os.path.join(root_path, "data", "user_dict.txt")

# 停⽤词路径和jieba分词⽤户⾃定义字典路径
train_seg_path = os.path.join(root_path, "data", "train_seg_data.csv")
test_seg_path = os.path.join(root_path, "data", "test_seg_data.csv")

# 预处理+切分后的训练测试数据路径
merged_seg_path = os.path.join(root_path, "data", "merged_seg_data.csv")

# 样本与标签分离，并经过pad处理后的数据路径
train_x_pad_path = os.path.join(root_path, "data", "train_x_pad_data.csv")
train_y_pad_path = os.path.join(root_path, "data", "train_y_pad_data.csv")
test_x_pad_path = os.path.join(root_path, "data", "test_x_pad_data.csv")

# numpy转换为数字后最终使⽤的的数据路径
train_x_path = os.path.join(root_path, 'data', 'train_X.npy')
train_y_path = os.path.join(root_path, 'data', 'train_Y.npy')
test_x_path = os.path.join(root_path, 'data', 'test_X.npy')

# 正向词典和反向词典路径
vocab_path = os.path.join(root_path, 'data', 'wv', 'vocab.txt')
reverse_vocab_path = os.path.join(root_path, 'data', 'wv', 'reverse_vocab.txt')

# 测试集结果保存路径
result_save_path = os.path.join(root_path, 'data', 'result')



