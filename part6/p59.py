from nltk import stem
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import optuna

#import nltk
#nltk.download('stopwords')

# 改良した特徴量を取得
def make_stemm_feature_file(input_file_name):
    # タイトル部分を読み込む
    df = pd.read_csv(input_file_name, sep='\t')
    title_input_file = df["title"]
    
    # 数字を統一して、ステミングを行う(語幹化)
    stemmer = stem.LancasterStemmer()
    stopWords = stopwords.words('english')
    for i in range(len(title_input_file)):
        title_input_file[i] = re.sub(r'\d+', '0', title_input_file[i])
        # 小文字化
        title_input_file[i] = title_input_file[i].lower()
        word_list = title_input_file[i].split()
        # ステミング
        for j in range(len(word_list)):
            word_list[j] = stemmer.stem(word_list[j])
        #ストップワード削除
        for elem in word_list:
            if elem in stopWords:
                word_list.remove(elem)
        title_input_file[i] = " ".join(word_list)

    # sklearnを用いてタイトルをtf-idfのベクトルにする
    vectorizer = TfidfVectorizer()
    feature = vectorizer.fit_transform(title_input_file)
    index_col = vectorizer.get_feature_names()
    

    # データフレームに変換し、ファイルに保存する
    # ベクトルはscipyの疎行列になっているので、numpy.array型に変換してからデータフレームにする
    all_feature_df = pd.DataFrame(feature.toarray(), columns=index_col)
    train_feature_df, test_valid_feature_df = train_test_split(all_feature_df, test_size=0.2, shuffle=False)
    valid_feature_df, test_feature_df = train_test_split(test_valid_feature_df, test_size=0.5, shuffle=False)
    train_feature_df.to_csv("train.feature.improved.txt", sep='\t')
    valid_feature_df.to_csv("valid.feature.improved.txt", sep='\t')
    test_feature_df.to_csv("test.feature.improved.txt", sep='\t')
    return

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

    pickle.dump(lr_model, open("lr_model_0.sav", 'wb'))
    return 

# 予測結果を得る
def predict_category(x_test):

    # 保存しといたモデルをロードして、特徴量を入力して予測クラスと確率を出力する
    lr_model = pickle.load(open("lr_model_0.sav", 'rb'))
    y_pred = lr_model.predict(x_test)
    return y_pred

# 正解率を得る
def output_model_accuracy(y_pred, y_test):
    # 保存しといたモデルをロードして、正解率を出力する
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    return acc




""" test_feature_df = pd.read_csv("valid.feature.improved.txt", sep='\t')
test_df = pd.read_csv("valid.txt", sep='\t')
y_test = test_df["category"]


train_logistic_regression("train.feature.improved.txt", "train.txt")
y_pred = predict_category(test_feature_df)
print(output_model_accuracy(y_pred, y_test)) """

'''
make_stemm_feature_file("all.txt")

tr_df = pd.read_csv("train.feature.improved.txt", sep='\t')
print(tr_df.shape)
te_df = pd.read_csv("test.feature.improved.txt", sep='\t')
print(te_df.shape)
va_df = pd.read_csv("valid.feature.improved.txt", sep='\t')
print(va_df.shape)
'''


# ロジスティック回帰モデルを学習する
def train_logistic_regression_with_c(c):
    # データを読み込む
    x_train = pd.read_csv("train.feature.improved.txt", sep='\t')
    x_df = pd.read_csv("train.txt", sep='\t')
    x_test = x_df["category"]
    y_train = pd.read_csv("valid.feature.improved.txt", sep='\t')
    y_df = pd.read_csv("valid.txt", sep='\t')
    y_test = y_df["category"]
    z_train = pd.read_csv("test.feature.improved.txt", sep='\t')
    z_df = pd.read_csv("test.txt", sep='\t')
    z_test = z_df["category"]

    # ロジスティック回帰モデルを学習させる, デフォルトはl2正則化
    lr_model = LogisticRegression(class_weight='balanced', C=c)
    lr_model.fit(x_train, x_test)

    # 予測を行い、正解率を求める
    x_pred = lr_model.predict(x_train)
    y_pred = lr_model.predict(y_train)
    z_pred = lr_model.predict(z_train)
    x_acc = accuracy_score(y_true=x_test, y_pred=x_pred)
    y_acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    z_acc = accuracy_score(y_true=z_test, y_pred=z_pred)
    return x_acc, y_acc, z_acc


# cを変えながら学習させ、正解率の推移をグラフで表示する
def output_c_accuracy_graph():
    x_acc_list = []
    y_acc_list = []
    z_acc_list = []
    index_list = []

    # 学習させて正解率の値をリストに格納する
    for i in range(1,11):
        c = i*0.1
        x_acc, y_acc, z_acc = train_logistic_regression(c)
        x_acc_list.append(x_acc)
        y_acc_list.append(y_acc)
        z_acc_list.append(z_acc)
        index_list.append(c)
    
    # グラフの描画
    plt.title("ofservation of overfitting")
    plt.xlabel("c")
    plt.ylabel("accuracy")
    plt.plot(index_list, x_acc_list, color="red")
    plt.plot(index_list, y_acc_list, color="blue")
    plt.plot(index_list, z_acc_list, color="green")
    plt.show()
    return


class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # ハイパーパラメータの設定
        params = {
            # 最適化アルゴリズムを指定
            'solver' : trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            # 正則化パラメータの範囲
            'C': trial.suggest_loguniform('C', 0.0001, 10),
            # 最大反復回数（＊ソルバーが収束するまで）
            'max_iter': trial.suggest_int('max_iter', 100, 10000)
            }

        model = LogisticRegression(**params)

        # 評価指標として正解率の最大化を目指す
        scores = cross_validate(model,
                                X=self.X, 
                                y=self.y,
                                scoring='accuracy',
                                n_jobs=-1)
        return scores['test_score'].mean()


def output_optimal_param(X_train, y_train):
    # ハイパーパラメータの探索
    objective = Objective(X_train, y_train)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=60)

    # ベストパラメータを出力
    print(study.best_params)
    print(study.best_value)
    return


valid_feature_df = pd.read_csv("valid.feature.improved.txt", sep='\t')
valid_df = pd.read_csv("valid.txt", sep='\t')
y_test = valid_df["category"]

#output_c_accuracy_graph()
#output_optimal_param(train_feature_df, y_train)

print(train_logistic_regression_with_c(c=0.2322599))


'''
実行結果
0.8863729433272395

(8756, 10277)
(1095, 10277)
(1094, 10277)
'solver': 'newton-cg', 'C': 0.23225992574717552, 'max_iter': 9363}
'''