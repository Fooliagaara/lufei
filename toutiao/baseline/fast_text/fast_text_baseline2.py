import fasttext
import time

train_data_path = 'data/train_fast.txt'
dev_data_path = 'data/dev_fast.txt'
test_data_path = 'data/test_fast.txt'

# autotuneValidationFile参数需要指定验证数据集所在路径,
# 它将在验证集上使用随机搜索方法寻找可能最优的超参数.
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300s,
# 根据不同的需求, 我们可以延长或缩短时间.
# verbose: 该参数决定日志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来.
# 日志打印等级
# 0：安静，几乎无输出
# 1：少量信息
# 2：默认，基础训练日志
# 3：最详细，打印自动调优每一组尝试的超参、精度、耗时，方便调试。
model = fasttext.train_supervised(input=train_data_path,
                                  autotuneValidationFile=dev_data_path,
                                  autotuneDuration=600,
                                  wordNgrams=2,
                                  verbose=3)

# 在测试集上评估模型的表现
result = model.test(test_data_path)
print(result)

# 模型保存
time1 = int(time.time())
model_save_path = "./toutiao_fasttext_{}.bin".format(time1)
model.save_model(model_save_path)