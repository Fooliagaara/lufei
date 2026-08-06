import fasttext

train_data_path = 'data/train_fast.txt'
test_data_path = 'data/test_fast.txt'

# 开启模型训练
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)

# 开启模型测试
result = model.test(test_data_path)
print(result)