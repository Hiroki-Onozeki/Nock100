from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# ロジスティック回帰モデルを学習する
def train_logistic_regression(c):
    # データを読み込む
    x_train = pd.read_csv("train.feature.txt", sep='\t')
    x_df = pd.read_csv("train.txt", sep='\t')
    x_test = x_df["category"]
    y_train = pd.read_csv("valid.feature.txt", sep='\t')
    y_df = pd.read_csv("valid.txt", sep='\t')
    y_test = y_df["category"]
    z_train = pd.read_csv("test.feature.txt", sep='\t')
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
    for i in range(4):
        c = i*0.01
        x_acc, y_acc, z_acc = train_logistic_regression(c)
        x_acc_list.append(x_acc)
        y_acc_list.append(y_acc)
        z_acc_list.append(z_acc)
        index_list.append(i)
    
    # グラフの描画
    plt.title("ofservation of overfitting")
    plt.xlabel("c")
    plt.ylabel("accuracy")
    plt.plot(index_list, x_acc_list, color="red")
    plt.plot(index_list, y_acc_list, color="blue")
    plt.plot(index_list, z_acc_list, color="green")
    plt.show()
    return

output_c_accuracy_graph()