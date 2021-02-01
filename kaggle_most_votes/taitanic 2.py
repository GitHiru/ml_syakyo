import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

train_raw = pd.read.csv('csv/train.csv')
test_raw = pd.read.csv('csv/test.csv')

train_mid = train_row.copy()
test_mid = train_row.copy()
train_mid['train_or_test'] = 'train'
test_mid['train_or_test'] = 'test'
test_mid['Survived'] = 9

alldata = pd.concat(
    [train_mid, test_mid],
    sort=False,
    axis=0).
    reset_index(drop=True
    )
