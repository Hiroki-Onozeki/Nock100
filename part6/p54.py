import pickle
import pandas as pd
import p53
from sklearn.metrics import accuracy_score

#　モデルの正解率を、予測結果と正解から求める
def output_model_accuracy(y_pred, y_test):
    # 保存しといたモデルをロードして、正解率を出力する
    lr_model = pickle.load(open("lr_model.sav", 'rb'))
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    return acc


# 正解のクラスと、特徴量を５３の関数に入れて予測クラスを得る
train_feature_df = pd.read_csv("train.feature.txt", sep='\t')
train_df = pd.read_csv("train.txt", sep='\t')
x_test = train_df["category"]
x_pred = p53.predict_category(train_feature_df)

test_feature_df = pd.read_csv("test.feature.txt", sep='\t')
test_df = pd.read_csv("test.txt", sep='\t')
y_test = test_df["category"]
y_pred = p53.predict_category(test_feature_df)

print(output_model_accuracy(x_pred, x_test))
print(output_model_accuracy(y_pred, y_test))


'''
実行結果

学習データ　0.8691183188670626
評価データ　0.8374429223744292
'''