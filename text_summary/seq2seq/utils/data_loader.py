import numpy as np
import os
import sys

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





