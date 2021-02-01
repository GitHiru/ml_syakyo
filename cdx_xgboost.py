import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metric import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

%matplotlib inline # plt.show()を省略

# (Kaggle) Human Resources Data Set
df = pd.read_csv('HRDataset_v14.csv', index_col=['Employ_Name'])
df = df[['Termd', 'Position', 'Sex', 'MaritalDesc', 'RaceDesc',
         'Department', 'ManagerName', 'RecruitmentSource',
         'EngagementSurvey','EmpSatisfaction', 'SpecialProjectsCount', ]]

# i) EDA (探索的データ解析)
df.head()
df.shape
df.isnull().sum()
df['Position'].value_counts()[0:10].plot(kind='pie', figsize=(4,4))
df['RecruitmentSource'].value_counts()[0:10].plot(kind='pie', figsize=(4,4))
df['Termd'].value_counts(dropna=False)

# ii) 前処理
dummy_cols = ['Position', 'Sex', 'MaritalDesc', 'RaceDesc',
              'Department', 'ManagerName', 'RecruitmentSource']
df = pd.get_dummies(df, columns=dummy_cols)    # str→int(0,1)
# df.shape

# iii) 目的変数 説明変数 分割
y = df['Termd']
x = df.drop(['Termd'], axis=1)    # Termd 以外全て説明変数(特徴量)
# y.shape, x.shape

# iv) 訓練データ テストデータ 分割
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
# x_train.shape, y_train.shape, x_test.shape, y_test.shape
# pd.count([y_train.value_counts(), y_test.value_counts()], axis=1)

# v) 予測検証
# 0. XGBoost （ベースライン策定）
params = {'metric':'error',
          'objective':'binary:logistic',
          'n_estimators':50000,
          'booster': 'gbtree',
          'learning_rate':0.01,
          'min_child_weight':1,
          'max_depth':5,
          'random_state':seed,
          'colsample_bytree':1,
          'subsample':1,
         }

cls = xgb.XGBClassifier()    # [instance]

cls.set_params(**params)
cls.fit(x_train,
        y_train,
        early_stopping_rounds=50,
        eval_set=[(x_test, y_test)],
        eval_metric='error',
        verbose=1)    # [learning]
pred_xgb = cls.predict(x_test)    # [predict]

# 評価 指標：正解率 混同行列
print('best score : ',cls.best_score)
print('best iterate : ',cls.best_iteration)
print('accuracy score :', accuracy_score(y_test, pred_xgb))
print('confusion matrix = \n', confusion_matrix(y_test, pred_xgb))


# 1. Grid Search
cv_params = {'metric':['error'],                 # 検証非対称
             'objective':['binary:logistic'],    # 検証非対称
             'n_estimators':[50000],
             'random_state':[seed],
             'booster': ['gbtree'],
             'learning_rate':[0.01],
             'min_child_weight':[1,5],           # 検証対称
             'max_depth':[1,3],                  # 検証対称
             'colsample_bytree':[0.5,1.0],       # 検証対称
             'subsample':[0.5,1.0]               # 検証対称
            }

cls_grid = GridSearchCV(cls,
                        cv_params,
                        cv=KFold(2, random_state=seed), scoring='accuracy',
                        iid=False)
cls_grid.fit(x_train,
             y_train,
             early_stopping_rounds=50,
             eval_set=[(x_test, y_test)],
             eval_metric='error',
             verbose=0)    # [learning]
pred_grid = cls_grid.best_estimator_.predict(x_test)   # [predict]

# 評価
print('best parameters : ', cls_grid.best_params_)    # 交差検証
print('best score : ', cls_grid.best_score_)
print('accuracy score :', accuracy_score(y_test, pred_grid))
print('confusion matrix = \n', confusion_matrix(y_test, pred_grid))


# 2. Random Search
cv_params = {'metric':['error'],
             'objective':['binary:logistic'],
             'n_estimators':[50000],
             'random_state':[seed],
             'boosting_type': ['gbdt'],
             'learning_rate':[0.01],
             'min_child_weight':[1,2,3,4,5,6,7,8,9,10],
             'max_depth':[1,2,3,4,5,6,7,8,9,10],
             'colsample_bytree':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
             'subsample':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
            }

cls_rdm = RandomizedSearchCV(cls,
                             cv_params,
                             cv=KFold(2, random_state=seed),
                             random_state=seed,
                             n_iter=30,
                             iid=False,
                             scoring='accuracy')
cls_rdm.fit(x_train,
            y_train,
            early_stopping_rounds=50,
            eval_set=[(x_test, y_test)],
            eval_metric='error',
            verbose=0)    # [learning]
pred_rdn = cls_rdm.best_estimator_.predict(x_test)    # [predict]

print('best parameters : ', cls_rdm.best_params_)
print('best score : ', cls_rdm.best_score_)
print('accuracy score :', accuracy_score(y_test, pred_rdm))
print('confusion matrix = \n', confusion_matrix(y_test, pred_rdm))


# 3. Bayesian Optimization (ベイズ最適化)
from bayes_opt import BayesianOptimization

def xgb_evaluate(min_child_weight, subsample, colsample_bytree, max_depth):
    params = {'metric': 'error',
              'objective':'binary:logistic',
              'n_estimators':50000,
              'random_state':42,
              'boosting_type':'gbdt',
              'learning_rate':0.01,
              'min_child_weight': int(min_child_weight),
              'max_depth': int(max_depth),
              'colsample_bytree': colsample_bytree,
              'subsample': subsample,
             }

    cls = xgb.XGBClassifier()
    cls.set_params(**params)
    cls.fit(x_train,
            y_train,
            early_stopping_rounds=50,
            eval_set=[(x_test, y_test)],
            eval_metric='error',
            verbose=0)    # [learning]

    pred = cls.predict(x_test)    # [predict]
    score = accuracy_score(y_test, pred)
    return score

# 検証実行
xgb_bo = BayesianOptimization(xgb_evaluate,
                              {'min_child_weight': (1,20),
                               'subsample': (.1,1),
                               'colsample_bytree': (.1,1),
                               'max_depth': (1,50)},
                               random_state=10)
xgb_bo.maximize(init_points=15, n_iter=50, acq='ei')

optimized_params = xgb_bo.max['params']    # max属性:最評価スコア結果取得
optimized_params['max_depth'] = int(optimized_params['max_depth'])

fixed_params = {'metric':'error',
                'objective':'binary:logistic',
                'n_estimators':50000,
                'random_state':seed,
                'booster': 'gbtree',
                'learning_rate':0.01}

cls.set_params(**fixed_params, **optimized_params)
cls.fit(x_train,
        y_train,
        early_stopping_rounds=50,
        eval_set=[(x_test, y_test)],
        eval_metric='error',
        verbose=0)    # [learning]
pred_bo = cls.predict(x_test)    # [predict]

print('best parameters : ', optimized_params)
print('accuracy score :', accuracy_score(y_test, pred_bo))
print('confusion matrix = \n', confusion_matrix(y_test, pred_bo))
