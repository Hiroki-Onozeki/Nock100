import pickle
import pandas as pd


# 記事見出しからカテゴリの予測確率を出力する
def predict_category(x_test):

    # 保存しといたモデルをロードして、特徴量を入力して予測クラスと確率を出力する
    lr_model = pickle.load(open("lr_model.sav", 'rb'))
    y_pred = lr_model.predict(x_test)
    y_prob = lr_model.predict_proba(x_test)

    # 予測確率とクラスの対応の確認
    #print(lr_model.classes_)
    #print(y_prob)
    return y_pred

'''
df = pd.read_csv("test.feature.txt", sep='\t')
x_test = df[1:2]

print(predict_category(x_test))
'''


'''
実行結果

['b' 'e' 'm' 't']
[[0.0083597  0.71394699 0.08239598 0.19529733]]
['e']
'''