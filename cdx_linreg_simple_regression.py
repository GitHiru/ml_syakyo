import numpy as np
import matplotlib.pyplot as plt

# 単回帰分析フルスクラッチ
data = np.loadtxt('/csv/007-01.csv', delimiter=',', skiprows=1)

# EDA データ確認
data.shape
plt.plot(data[:, 0], [:,1], 'bx', markersize=10, label='Traning')
plt.xlabel('Test Score')
plt.ylabel('Study Hour')
plt.grid(True)
plt.show()

# コスト関数
def cost(w0, w1, data):
    cost = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        cost += (((w1*x)+w0)-y)**2
    cost = cost/(len(data)*2)
    return cost

# 最急降下法（Gradient Descent）
def gradientDescent(w0_in, w1_in, data, alpha):
    w0_gradient = 0
    w1_gradient = 0
    m = float(len(data))
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        w0_gradient += (1/m)*(((w1_in*x)+w0_in)-y)
        w1_gradient += (1/m)*((((w1_in*x)+w0_in)-y)*x)
    w0_out = w0_in-(alpha*w0_gradient)
    w1_out = w1_in-(alpha*w1_gradient)
    return [w0_out, w1_out]

# 単回帰分析 実行
def run(data, init_w0, init_w1, alpha, iterations):
    w0 = init_w0
    w1 = init_w1
    for i in range(iterations):
        w0, w1 = gradientDescent(w0, w1, np.array(data), alpha)
    return [w0, w1]

# ConfiGlation
init_w0 = 0
init_w1 = 0
alpha = 0.01
iterations = 5000
[w0, w1] = run(init_w0, init_w1, alpha, iterations)
print([w0, w1])

# 線形図
plt.plot(data[:, 0], data[:,0]*w1+w0, 'r-', label='Linear Regression')
# 散布図
plt.plot(data[:, 0], data[:, 1], 'rx', markersize=10, label='Traning')
plt.xlabel('Test Score')
plt.ylabel('Study Hour')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# 3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm     # color mat

w0_vals = np.linspace(-75, 80, 100)
w1_vals = np.linspace(-75, 80, 100)
J_vals = np.zeros((len(w0_vals), len(w1_vals)))
for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        J_vals[i,j] = cost(w0_vals[i], w1_vals[j], data)
J_vals = np.transpose(J_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
w0_g, s1_g = np.meshgrid(w0_vals, w1_vals)     # necessary for 3D graph
surf = ax.plot_surface(w0_g, w1_g, J_vals, cmap=cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('w0')
plt.ylabel('w1')
plt.show()
