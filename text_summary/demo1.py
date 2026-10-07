import pandas as pd

train_path = 'data/train.csv'
test_path = 'data/test.csv'

df = pd.read_csv(train_path, encoding='utf-8')
df.info()

print("*****************************************")

df = pd.read_csv(test_path, encoding='utf-8')
df.info()


