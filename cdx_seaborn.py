import pandas as pd
import numpy as np
from skleran.preprocessiong import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

train_mid = train_raw.copy()
test_mid = test_raw.copy()
train_mid['train_or_test'] = 'train'
test_mid['train_or_test'] = 'test'
test_mid['Survived'] = 9
alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)

# print('The size of the train data :', train_raw.shape)
# print('The size of the test data :', test_raw.shape)
# print('The size of the alldata data :', alldata.shape)

# 前処理
alldata.isnull().sum()
alldata.Embarked.fillna(alldata.Embarked.mode()[0], inplace=True)
alldata.Fare.fillna(alldata.Fare.median(), inplace=True)
