# coding=utf-8
# 导⼊textrank4zh的相关⼯具包
from textrank4zh import TextRank4Keyword, TextRank4Sentence
# 导⼊常⽤⼯具包
import pandas as pd
import numpy as np
import jieba.analyse

#关键词抽取
def keywords_extraction(text):
    # allow_speech_tags : 词性列表, ⽤于过滤某些词性的词
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt',
    'nz'])
    # text: ⽂本内容, 字符串
    # window: 窗⼝⼤⼩, int, ⽤来构造单词之间的边, 默认值为2
    # lower: 是否将英⽂⽂本转换为⼩写, 默认值为False
    # vertex_source: 选择使⽤words_no_filter, words_no_stop_words,words_all_filters中的>哪⼀个来构造pagerank对应的图中的节点
    # 默认值为'all_filters', 可选值为'no_filter', 'no_stop_words','all_filters'
    # edge_source: 选择使⽤words_no_filter, words_no_stop_words, words_all_filters中的哪>⼀个来构造pagerank对应的图中的节点之间的边
    # 默认值为'no_stop_words', 可选值为'no_filter', 'no_stop_words','all_filters', 边的构造要结合window参数
    # pagerank_config: pagerank算法参数配置, 阻尼系数为0.85
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters',
    edge_source='no_stop_words', pagerank_config={'alpha': 0.85, })
    # num: 返回关键词数量
    # word_min_len: 词的最⼩⻓度, 默认值为1 
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    # 返回关键词
    return keywords

#关键短语抽取
def keyphrases_extraction(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters',
    edge_source='no_stop_words', pagerank_config={'alpha': 0.85, })
    # keywords_num: 抽取的关键词数量
    # min_occur_num: 关键短语在⽂中的最少出现次数
    keyphrases = tr4w.get_keyphrases(keywords_num=6, min_occur_num=1)
    # 返回关键短语
    return keyphrases

#关键句抽取
def keysentences_extraction(text):
    tr4s = TextRank4Sentence()
    # text: ⽂本内容, 字符串
    # lower: 是否将英⽂⽂本转换为⼩写, 默认值为False
    # source: 选择使⽤words_no_filter, words_no_stop_words, words_all_filters中的哪⼀个来⽣成句⼦之间的相似度
    # 默认值为'all_filters', 可选值为'no_filter', 'no_stop_words','all_filters'
    tr4s.analyze(text, lower=True, source='all_filters')
    # 获取最重要的num个⻓度⼤于等于sentence_min_len的句⼦⽤来⽣成摘要
    keysentences = tr4s.get_key_sentences(num=3, sentence_min_len=6)
    # 返回关键句⼦
    return keysentences

def jieba_keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords


if __name__ == "__main__":
    text =  "来源：中国科学报本报讯（记者肖洁）又有⼀位中国科学家喜获小行星命名殊荣！4月19日下午，中国科学院国家天文台在京举行“周又元星”颁授仪式，" \
    "我国天文学家、中国科学院院士周又元的弟子与后辈在欢声笑语中济济⼀堂。国家天文台党委书记、" \
    "副台长赵刚在致辞⼀开始更是送上白居易的诗句：“令公桃李满天下，何须堂前更种花。”" \
    "据介绍，这颗小行星由国家天文台施密特CCD小行星项目组于1997年9⽉26⽇发现于兴隆观测站，" \
    "获得国际永久编号第120730号。2018年9⽉25⽇，经国家天文台申报，" \
    "国际天文学联合会小天体联合会小天体命名委员会批准，国际天文学联合会《小行星通报》通知国际社会，" \
    "正式将该小行星命名为“周又元星”。"
    
    #关键词抽取
    # keywords=keywords_extraction(text)
    keyphrases=keyphrases_extraction(text)
    # keysentences = keysentences_extraction(text)
    # keywords = jieba_keywords_textrank(text)
    print(keyphrases)