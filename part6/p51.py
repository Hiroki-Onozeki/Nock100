from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# タイトルをtf-idfによって抽出した特徴量にしたものをファイルにする
def make_feature_file(input_file_name):
    
    df = pd.read_csv(input_file_name, sep='\t')
    title_input_file = df["title"]

    # sklearnを用いてタイトルをtf-idfのベクトルにする
    vectorizer = TfidfVectorizer()
    feature = vectorizer.fit_transform(title_input_file)
    index_col = vectorizer.get_feature_names()
    

    # データフレームに変換し、ファイルに保存する
    # ベクトルはscipyの疎行列になっているので、numpy.array型に変換してからデータフレームにする
    all_feature_df = pd.DataFrame(feature.toarray(), columns=index_col)
    train_feature_df, test_valid_feature_df = train_test_split(all_feature_df, test_size=0.2, shuffle=False)
    valid_feature_df, test_feature_df = train_test_split(test_valid_feature_df, test_size=0.5, shuffle=False)
    train_feature_df.to_csv("train.feature.txt", sep='\t')
    valid_feature_df.to_csv("valid.feature.txt", sep='\t')
    test_feature_df.to_csv("test.feature.txt", sep='\t')
    
    return


make_feature_file("all.txt")


tr_df = pd.read_csv("train.feature.txt", sep='\t')
print(tr_df.shape)
te_df = pd.read_csv("test.feature.txt", sep='\t')
print(te_df.shape)
va_df = pd.read_csv("valid.feature.txt", sep='\t')
print(va_df.shape)


'''
実行結果

(8756, 12866)
(1095, 12866)
(1094, 12866)
'''