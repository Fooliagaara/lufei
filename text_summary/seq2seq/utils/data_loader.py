import numpy as np
import os
import sys
import re
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def get_max_len(data):
    # 获得合适的最⼤⻓度值(被build_dataset调⽤)
    # data: 待统计的数据train_df['Question']
    # 句⼦最⼤⻓度为空格数+1
    max_len = data.apply(lambda x: x.count(' ') + 1)
    # 平均值+2倍方差的方式
    return int(np.mean(max_len) + 2 * np.std(max_len))


def transform_data(sentence, word_to_id):
    # 句⼦转换为index序列(被build_dataset调⽤)
    # sentence: 'word1 word2 word3 ...' -> [index1, index2, index3 ...]
    # word_to_id: 映射字典

    # 字符串切分成词
    words = sentence.split(' ')

    # 按照word_to_id的id进行转换，到未知词就填充unk的索引
    ids = [word_to_id[w] if w in word_to_id else word_to_id['unk'] for w in words]

    # 返回映射后的文本id值列表
    return ids


def pad_proc(sentence, max_len, word_to_id):
    # 根据max_len和vocab填充<START> <STOP> <PAD> <UNK>

    words = sentence.strip().split(' ')

    words = words[:max_len]

    sentence = [w if w in word_to_id else '<UNK>' for w in words]

    sentence = ['<START>'] + sentence + ['<STOP>']

    sentence = sentence + ['<PAD>'] * (max_len - len(words))

    return ' '.join(sentence)


def load_stop_words(stop_word_path):
    # 加载停⽤词(程序调⽤)
    # stop_word_path: 停⽤词路径
    # 打开停⽤词⽂件
    f = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = f.readlines()

    # 去除每一个停用词前后的空格和换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


def clean_sentence(sentence):
    # 特殊符号去除(被sentence_proc调⽤)
    # sentence: 待处理的字符串
    if isinstance(sentence, str):
        # 删除1. 2. 3. 这些标题
        r = re.compile("\D(\d\.)\D")
        sentence = r.sub("", sentence)
        # 删除带括号的 进⼝ 海外
        r = re.compile(r"[(（]进⼝[)）]|\(海外\)")
        sentence = r.sub("", sentence)
        # 删除除了汉字数字字⺟和，！？。.- 以外的字符
        r = re.compile(r"[^，！？。\.\-\u4e00-\u9fa5_a-zA-Z0-9\u2e80-\u2eff\u2f00-\u2fdf]")
        # ⽤中⽂输⼊法下的，！？来替换英⽂输⼊法下的,!?
        sentence = sentence.replace(",", "，")
        sentence = sentence.replace("!", "！")
        sentence = sentence.replace("?", "？")
        sentence = r.sub("", sentence)
        # 删除 ⻋主说 技师说 语⾳ 图⽚ 你好 您好
        r = re.compile(r"⻋主说|技师说|语⾳|图⽚|你好|您好")
        sentence = r.sub("", sentence)
        return sentence
    else:
        return ''


def filter_stopwords(seg_list, stop_words):
    # 过滤⼀句切好词的话中的停⽤词(被sentence_proc调⽤)
    # seg_list: 切好词的列表 [word1 ,word2 .......]
    # ⾸先去掉多余空字符
    words = [word for word in seg_list if word]
    # 去掉停⽤词
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence, stop_words):
    # 预处理模块(处理⼀条句⼦, 被sentences_proc调⽤)
    # sentence: 待处理字符串

    # 第⼀步: 执⾏清洗原始⽂本的操作
    sentence = clean_sentence(sentence)
    # 第⼆步: 执⾏分词操作, 默认精确模式, 全模式cut参数cut_all=True
    words = jieba.cut(sentence, cut_all=False)
    # 第三步: 将分词结果输⼊过滤停⽤词函数中
    words = filter_stopwords(words, stop_words)
    # 返回处理后的词列表
    return ' '.join(words)


def sentences_proc(df):
    # 预处理模块(处理⼀个句⼦列表, 对每个句⼦调⽤sentence_proc操作)
    # df: 数据集

    # 批量预处理训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    # 训练集Report预处理
    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)

    # 以Pandas的DataFrame格式返回
    return df








