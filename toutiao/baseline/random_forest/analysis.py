import pandas as pd
from collections import Counter
import numpy as np
import jieba


content = pd.read_csv('../../data/data/train.txt', sep='\t')

print(content.head(10))

print(len(content))

count = Counter(content.label.values)

print(count)
print(len(count))
print('***************************************')

total = 0
for i, v in count.items():
    total += v

print(total)

for i, v in count.items():
    print(i, v / total * 100, '%')

print('***************************************')

content['sentence_len'] = content['sentence'].apply(len)

print(content.head(10))

length_mean = np.mean(content['sentence_len'])
length_std = np.std(content['sentence_len'])
print('length_mean = ', length_mean)
print('length_std = ', length_std)


def cut_sentence(s):
    return list(jieba.cut(s))


content['words'] = content['sentence'].apply(cut_sentence)

print(content.head(10))

# content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))
#
# content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:30])

content.to_csv('../../data/train_new.csv')
