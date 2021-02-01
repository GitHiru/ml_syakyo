import matplotlib.pylot as plt
import numpy as np

# DATASET
g = np.random.randn(1000)    # Standard Nomal Distribustion
# DATAPLOT
plt.hist(g, bins=100)    # cumulative=True:累積値 range=():レンジ
plt.show()

# DATASET
n = np.random.random(1000)
# DATASET
labels = ['Gaussian', 'Continud Uniform Distribustion']
plt.hist(g, bins=100, alpha=0.3, color='red')
plt.hist(n, bins=100, alpha=0.3, color='blue')
plt.show()
