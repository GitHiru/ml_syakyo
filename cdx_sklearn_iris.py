import sklearn
import pandas as pd
from sklearn.datasets import load_iris

# 分類
iris = load_iris()

# 特徴量
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_name)
# iris_fertures.head()
# iris_featuers.describe()
# iris_features.isnull().sum()

# ラベル確認
iris_label = pd.Series(iris.target)
iris_label.value_counts()

# データ分割
from sklearn.model_selectiion import train_test_split
features_train, features_test, label_train, label_test = train_test_split(iris_features, iris_label, test_size=0.5, random_state=0)
# print(features_train.shape, label_train.shape, features_test.shape, label_test.shape)


# 1. LinearSVC model
from sklearn import svm
Linsvc = svm.LinearSVC(random_state=0, max_iter=3000)    #instance
Linsvc.fit(features_train, label_train)    # Learning
label_pred_linsvc = Linsvc.predict(features_test)    # Predict

# 2. KNeighbors model
from sklearn.neighors import KNeighborsClassifier
Kneighbor = KNeighborsClassifier(n_neighbors=5)
Kneighbor.fit(features_train, label_train)
label_pred_KNeighbor = Kneighbor.predict(features_test)

# 3. LogisticRegression model
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state=0)
LogReg.fit(feature_train, label_train)
label_pred_LogReg = LogReg.predict(features_test)


# 評価 指標:混同行列 正解率
from sklearn.metric import confusion_matrix
# 1. LinearSVC
print('confusion matrix = \n', confusion_matrix(y_true=label_test, y_pred=label_pred_linsvc))
print('accuracy = ', Linsvc.score(feature_test, label_test))
# 2. KNeighbors model
print('confusion matrix = \n', confusion_matrix(y_true=label_test, y_pred=label_pred_KNeighbor))
print('accuracy = ', Kneighbor.score(feature_test, label_test))
# 3. LogisticRegression model
print('confusion matrix = \n', confusion_matrix(y_true=label_test, y_pred=label_pred_LogReg))
print('accuracy = ', LogReg.score(feature_test, label_test))
