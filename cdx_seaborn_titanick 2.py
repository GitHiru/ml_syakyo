import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib.inline

df = sns.load_dataset('titanic')

# EDA
# df.head()
# df.isnull().sum()

sns.catplot(x='pclass', data=df, kind='count')     # bar chart
sns.distplot(df['age'], kde=False)     # histgram
sns.catplot(x='sex', y='age', data=df, kind='box', hue='survived') # hue:x,yのデータhueで指定データで分けてplot kund:boxen より詳細に
sns.catplot(x='sex', y='age', data=df, kind='bar', hue='survived')    # bar chart
sns.catplot(x='sex', y='age', data=df)    # scatter plot
sns.catplot(x='sex', y='age', data=df, kind='swarm')    # scatter plot
sns.catplot(x='sex', y='age', data=df, kind='swarm', hue='survived')    # scatter plot
sns.catplot(x='sex', y='age', data=df, kind='violin', hue='survived')    # violin plot
sns.catplot(x='sex', y='age', data=df, kind='violin', hue='survived', split=True)    # violin plot
sns.scatterplot(x='age', y='fare', data=df)    # scatter plot
sns.jointplot(x='age', y='fare', data=df, kind='hex')     # hist+sccater hex:データ濃淡


df2 = sns.load_dataset('iris')
df2.head()
sns.pairplot(df2)    # many plots
sns.pairplot(data=df2, hue='species')     # many plots
