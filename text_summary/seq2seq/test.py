import os
import sys
import jieba
from utils.config import *
from utils.data_loader import load_stop_words, clean_sentence, filter_stopwords, sentence_proc


# # 加载停⽤词, 这⾥面的stop_words_path是早已在config.py⽂件中配置好的
stop_words = load_stop_words(stop_words_path)
print('stop_words: ', stop_words[:20])


sentence = "技师说：你好！以前也出现过该故障吗？|技师说：缸压多少有没有测量⼀下?|⻋主说：没有过|⻋主说：没测缸压|技师说：测量⼀下缸压 看⼀四缸缸压是否偏低|⻋主说：⽤电脑测，只是14缸缺⽕|⻋主说：[语⾳]|⻋主说：[语⾳]|技师说：点⽕线圈 ⽕花塞 喷油嘴不⽤⼲活 直接和⼆三缸对倒⼀下 跑⼀段在测量⼀下故障码进⾏排除|⻋主说：[语⾳]|⻋主>说：[语⾳]|⻋主说：[语⾳]|⻋主说：[语⾳]|⻋主说：师傅还在吗|技师说：调⼀下喷油嘴 测⼀下缸压 都正常则为发动机电脑板问题|⻋主说：[语⾳]|⻋主说：[语⾳]|⻋主说：[语⾳]|技师说：这个影响不⼤的|技师说：缸压⼋个以上正常|⻋主说：[语⾳]|技师说：所以说让你测量缸压 只要缸压正常则没有问题|⻋主说：[语⾳]|⻋主说：[语⾳]|技师说：可以点击头像关注我 有什么问题随时询问 ⼀定真诚⽤心为你解决|⻋主说：师傅，谢谢了|技师说：不⽤客⽓"

# res = clean_sentence(sentence)
# print('res: ', res)

# words = jieba.cut(res)

# result = filter_stopwords(words, stop_words)
# print('result: ', result)

res = sentence_proc(sentence, stop_words)
print('res: ', res)


