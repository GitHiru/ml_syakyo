import sklearn
import pandas as pd
from sklearn.dataset import load_boston

# 回帰
boston = load_boston()

# 特徴量
boston_features = pd.DataFrame(data=boston.data, columns=boston.feature_name)
# boston_features.head()
# boston_features.describe()
# boston_features.isnull().sum()

boston_target = pd.Serise(data=boston.target)
# boston_target.describe()

# データ分割
from sklearn.model_selection import train_test_split
features_train,
features_test,
target_train,
target_test = train_test_split(boston_features, boston_target, test_size=0.5, random_state=0)

# 1. Lasso ラッソ回帰
from skleaarn.linear_model import Lasso
Lasso = Lasso(alpha=0.1, random_state=0)    # instance
Lasso.fit(features_train, target_train)    # learning
target_pred_Lasso = Lasso.predict(features_test)    # predict

# 2. Ridge Regression リッジ回帰
from skleaarn.linear_model import Ridge
Ridge = Ridge(alpha=0.5, random_state=0)
Ridge.fit(features_train, target_train)
target_pred_Ridge = Ridge.predict(features_test)

# cf. 3. Linear Regression 線形回帰
from skleaarn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(features_train, target_train)
target_pred_LinReg = LinReg.predict(features_test)


# 評価1 指標:決定係数
# 1. Lasso モデルによる回帰の評価（決定係数の表示）
print('R-squared : ', Lasso.score(features_test, target_test))
# 2. Ridge Regression
print('R-squared : ', Ridge.score(features_test, target_test))
# Cf. 3. Linear Regression
print('R-squared : ', LinReg.score(features_test, target_test))


# 評価2 指標:Cross Validation 交差検証
from sklearn.model＿selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_split=5, test_size=0.2, random_state=0)
# 1. Lasso モデルによる回帰の評価（決定係数の表示）
score = cross_val_score(Lasso, boston_features, boston_target, cv=cv)
print('R-squared Average : {0:.2f}'.format(score.mean()))
# 2. Ridge Regression
score = cross_val_score(Ridge, boston_features, boston_target, cv=cv)
print('R-squared Average : {0:.2f}'.format(score.mean()))
# Cf. 3. Linear Regression
score = cross_val_score(LinReg, boston_features, boston_target, cv=cv)
print('R-squared Average : {0:.2f}'.format(score.mean()))
