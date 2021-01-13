import as matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# DATASET
df = pd.read_csv('/csv/sample.csv')
data = df.values
# EDA
df.shape
df.head()
type(data)

x = data[:,0]
y = data[:,1]

ans_x = np.linspace(0, 100, 100)
ans_y = 2 * ans_x + 12

# DATAPLOT
plt.scatter(x, y)
plt.plot(ans_x, ans_y, color='r', linewidth=3)
plt.show()
