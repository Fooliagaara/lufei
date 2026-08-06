import requests
import time

# 定义请求url和传入的data
url = "http://127.0.0.1:5000/v1/main_server/"
data = {"uid": "AI-6-202104", "text": "女主人成功说服杀人躲避者自首"}

start_time = time.time()
# 向服务发送post请求
res = requests.post(url, data=data)

cost_time = time.time() - start_time

# 打印返回的结果
print('输入文本:', data['text'])
print('分类结果:', res.text)
print('单条样本预测耗时: {:.2f}'.format(cost_time * 1000), 'ms')