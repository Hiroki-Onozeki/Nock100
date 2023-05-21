import pickle
import pandas as pd
import p53
from sklearn.metrics import confusion_matrix


# モデルの混同行列を、予測結果と正解から求める
def output_model_confusion_matrix(y_pred, y_test):
    # 保存しといたモデルをロードして、正解率を出力する
    lr_model = pickle.load(open("lr_model.sav", 'rb'))
    con_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return con_mat

'''
# 正解のクラスと、特徴量を５３の関数に入れて予測クラスを得る
train_feature_df = pd.read_csv("train.feature.txt", sep='\t')
train_df = pd.read_csv("train.txt", sep='\t')
x_test = train_df["category"]
x_pred = p53.predict_category(train_feature_df)

test_feature_df = pd.read_csv("test.feature.txt", sep='\t')
test_df = pd.read_csv("test.txt", sep='\t')
y_test = test_df["category"]
y_pred = p53.predict_category(test_feature_df)

print(output_model_confusion_matrix(x_pred, x_test))
print(output_model_confusion_matrix(y_pred, y_test))
'''

'''
実行結果

[[2542  267   79  194]
 [  81 3794   90  140]
 [  23   43  537   21]
 [  76   96   36  737]]
[[297  17   6  36]
 [ 12 489  13  33]
 [  6  14  54   7]
 [ 19  10   5  77]]
'''