import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DATASET
# Kaggle MNIST: https://www.kaggle.com/c/digit-recognizer/data?select=train.csv
df = pd.read_csv('csv/train.csv')

# EDA
df.shape
df.head()

# DATAPLOT
A = df.values    # pd→np
A1 = A[5, 1:]    # row:5 col:1~all
A1 = A1.reshape(28, 28)
plt.imshow(A1, cmap='gray')    # image show

A2 = A[10, 1:]
A2 = A2.reshape(28, 28)
plt.imshow(A2, cmap='gray')
plt.show()

#
# A1 = A[row, 1:]
# A1 = A1.reshape(28, 28)
# plt.imshow(A1, cmap='gray')
# plt.show()
