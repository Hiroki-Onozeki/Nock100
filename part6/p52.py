from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# ロジスティック回帰モデルを学習する
def train_logistic_regression(train_feature_file, train_data_file):
    feature_df = pd.read_csv(train_feature_file, sep='\t')
    all_df = pd.read_csv(train_data_file, sep='\t')

    # タイトルの特徴ベクトルを説明変数、カテゴリーをクラスとする
    x_train = feature_df
    y_train = all_df["category"]

    # ロジスティック回帰モデルを学習させる, 不均衡データなのでe,bしか予測しなくなってしまったので、重みを調整する
    lr_model = LogisticRegression(class_weight='balanced')
    lr_model.fit(x_train, y_train)

    # 切片と係数を確認する、モデルを保存しておく
    print(lr_model.intercept_, lr_model.coef_)
    pickle.dump(lr_model, open("lr_model.sav", 'wb'))
    return 

train_logistic_regression("train.feature.txt", "train.txt")

'''
実行結果　b, e, m, t

切片    [-0.30275179  0.68711983 -0.15169073 -0.2326773 ]
係数    [[ 4.17430018e-05  6.00249042e-02 -1.99173515e-03 ... -9.62705619e-03
   0.00000000e+00 -9.16564169e-03]
 [-4.37543411e-05 -2.02542129e-02  5.98138523e-03 ...  2.92058566e-02
   0.00000000e+00  2.77716965e-02]
 [-2.50285365e-05 -1.98910174e-02 -1.98894314e-03 ... -9.77425218e-03
   0.00000000e+00 -9.24732738e-03]
 [ 2.70399254e-05 -1.98796738e-02 -2.00070694e-03 ... -9.80454824e-03
   0.00000000e+00 -9.35872740e-03]]
'''