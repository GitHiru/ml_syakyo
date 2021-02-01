import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

# 重回帰分析フルスクラッチ
# i)データセット
data = np.loadtxt('/csv/sample.csv', delimiter=',', skiprows=1)

x = data[:, :3]    # (特徴量) col 0~3
y = data[:, 3]    # (Target) col 4 only
m = len(y)    # データ量


# ii)EDA(探索的データ分析)
data.shape

# データセットされたか確認
for i in rage(10):
    print('x = [{:.0f}{:.0f}{:.0f}], y = {:.0f}' .format(x[i,0], x[i,1], x[i,2], y[i]))

# 現状データ可視化（４次元はできないので、三次元で）
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,0], data[:,1], data[:,3], color='#ef1234')
ax.set_xlabel('Pysic')
ax.set_ylabel('Science')
ax.set_zlabel('Math')    # Target
Plt.show()


# iii) 正規化関数 (標準化：Z-score Nomalization 手法を用いる)
def norm(z):
    z_norm = np.zeros((z.shape[0], z.shape[1]))
    mean, std = np.zeros((1, z.shape[1]))

    for i in range(z.shape[1]):
        mean[:,i] = np.mean(z[:,i])
        std[:,i] = np.std(z[:,i])
        z_norm[:,i] = (z[:,i])-float(mean[:,i])/float(std[:,i])
    return z_norm


# iv) モデル設定（ベクトル化）
# X行列定義：正規化
X_norm = norm(x)
X_padded = np.column_stack((np.ones((m,1)), X_norm))

# W（パラメーター）ベクトル定義：初期値
weight_init = np.zeros((4,1))


# v) コスト関数(※ベクトル化前)
def cost(X, y, W):
    m = len(y)
    J = 0
    y_hut = X.dot(W)
    diff = np.power((y_hut - np.transpose([y])), 2)
    J = (1.0/(2*m)) * diff.sum(axis=0)
    return J


# vi) 最急降下法（Gradient Disent）
def gradientDescent(X, y, weight, alpha, iterations):
    m = len(y)
    J_history = np.zeros((iterations, 1))    #(500, 1)
    for i in range(iterarions):
        W = weight - alpha*(1.0/m)*transpose(X).dot(X.dot(weight) - np.transpose([y]))
        J_history[i] = cost(X, y, W)
    return W, J_history


# vii) 最適パラメーター取得
alpha = 0.01
num_itrs = 500

# 実行
W, J_history = gradientDescent(X_padded, y, weight_init, alpha, num_itrs)

# 最適パラメーター確認
for i in range(weight.size):
    print('w{} = {}' .format(i, weight[i].round(3)))
# w0:1
# w1:Pysics
# w2:Sciencs
# w3:Statistics

for i in range(J_history.size):
    print('最適化 {}回目: cost = {}' .format(i, J_history[i]).round(1))

# コストと学習回数のグラフ
plt.plot(range(J_history.size), J_history, '-b', linewidth=1)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost(J)')
plt.grid(True)
plt.show()

# 数学点数予測 = （物理点数＊w１）＋（化学点数＊w２）＋（統計学点数＊w３）＋w０
# ^^^^^^^^^^^^ (モデル完成) ^^^^^^^^^^^^^^^


# viii) 予測
# e.g  N君(物理：76点 化学:96点 統計:82点 数学:?点）の数学得点を学習モデルを用いて予測せよ。    (hint)得点を正規化して、ベクトル作成
pysics_norm = (76-float(mean[:,0])) /float(std[:,0])
science_norm = (96-float(mean[:,1])) /float(std[:,1])
statistics_norm = (82-float(mean[:,2])) /float(std[:,2])

pred_padded = np.array([1, pysics_norm, sciencs_norm, statistics_norm])

pred = pred_padded.dot(W)
print(pred)
