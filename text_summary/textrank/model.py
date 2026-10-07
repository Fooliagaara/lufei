import pandas as pd
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import re

def clean_sentence(sentence):
    sub_jishi = []
    sub = sentence.split("|")

    for i in range(len(sub)):
        if not sub[i].endswith(('。', '？', '！', '.', '?', '!')):
            sub[i] += '。'
        if sub[i].startswith('技师'):
            sub_jishi.append(sub[i])

    sentence = ''.join(sub_jishi)

    r = re.compile("\D(\d\.)\D")
    sentence = r.sub("", sentence)

    r = re.compile(r"⻋主说|技师说|语⾳|图⽚|呢|吧|哈|啊|啦")
    sentence = r.sub("", sentence)

    r = re.compile(r"[(（]进⼝[)）]|\(海外\)")
    sentence = r.sub("", sentence)
    # 5. 删除除了汉字数字字⺟和，！？。.- 以外的字符
    r = re.compile("[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9]")
    # 6. 半⻆变为全⻆
    sentence = sentence.replace(",", "，")
    sentence = sentence.replace("!", "！")
    sentence = sentence.replace("?", "？")
    # 7. 问号叹号变为句号
    sentence = sentence.replace("？", "。")
    sentence = sentence.replace("！", "。")
    sentence = r.sub("", sentence)

    # 第四步添加的删除特定位置的特定字符
    # 8. 删除句⼦开头的逗号
    if sentence.startswith('，'):
        sentence = sentence[1:]

    return sentence

 

if __name__ == '__main__':
    # 数据处理
    df = pd.read_csv('../data/dev.csv', encoding='utf-8')
    texts = df['Dialogue'].tolist()
    for i in range(len(texts)):
        texts[i] = clean_sentence(texts[i])
        if i % 500 == 0:
            print('i=', i)

    # 初始化结果存放的列表
    results = []
    # 初始化textrank4zh类对象
    tr4s = TextRank4Sentence()
    # 循环遍历整个测试集, texts是经历前⾯数据预处理后的结果列表
    for i in range(len(texts)):
        text = texts[i]
        # 直接调⽤分析函数
        tr4s.analyze(text=text, lower=True, source='all_filters')
        result = ''
        # 直接调⽤函数获取关键语句
        # num=3: 获取重要性最⾼的3个句⼦.
        # sentence_min_len=2: 句⼦的⻓度最⼩等于2.
        for item in tr4s.get_key_sentences(num=3, sentence_min_len=2):
            result += item.sentence
            result += '。'

        results.append(result)
        # 间隔100次打印结果
        if (i + 1) % 100 == 0:
            print(i + 1, result)
    print('result length: ', len(results))

    # 保存结果
    df['Prediction'] = results
    # 提取ID, Report, 和预测结果这3列
    df = df[['QID', 'Report', 'Prediction']]
    # 保存结果，这⾥⾃动⽣成⼀个结果名
    df.to_csv('../data/textrank_result_.csv', index=None, sep=',')
    # 将空⾏置换为随时联系, ⽂件保存格式指定为utf-8
    df = pd.read_csv('textrank_result_.csv', engine='python', encoding='utf-8')
    df = df.fillna('随时联系。')
    # 将处理后的⽂件保存起来
    df.to_csv('../data/textrank_result_final_.csv', index=None, sep=',')
