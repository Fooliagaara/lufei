import time
import jieba
import fasttext

# 服务框架使用Flask, 导入工具包
from flask import Flask
from flask import request

from waitress import serve
app = Flask(__name__)

# 导入发送http请求的requests工具
import requests

# 加载自定义的停用词字典
jieba.load_userdict('../../data/stopwords.txt')

# 提供已训练好的模型路径+名字
model_save_path = 'toutiao_fasttext_1785384163.bin'

# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print('FastText模型实例化完毕...')


# 设定投满分服务的路由和请求方法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 接收来自请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']

    # 对请求文本进行处理, 因为前面加载的是基于词的模型, 所以这里用jieba进行分词
    input_text = ' '.join(jieba.lcut(text))

    # 执行预测
    res = model.predict(input_text)
    predict_name = res[0][0]
    # print(res)

    return predict_name


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)