import pickle
import pandas as pd
import p55
import p53



# モデルの適合率・再現率・f１を、混同行列から求める
def output_model_precision_recall_f1(con_mat):
    precision_list = []
    recall_list = []
    f1_list = []

    # カテゴリごとの適合率を求めてリストに格納
    for i in range(4):
        totall = 0
        for j in range(4):
            totall += con_mat[j][i]
        precision_list.append(con_mat[i][i]/totall)

    # カテゴリごとの再現率を求めてリストに格納
    for i in range(4):
        totall = 0
        for j in range(4):
            totall += con_mat[i][j]
        recall_list.append(con_mat[i][i]/totall)

    # カテゴリごとのf1を求めてリストに格納
    for i in range(4):
        f1_score = (2*precision_list[i]*recall_list[i]) / (precision_list[i]+recall_list[i])
        f1_list.append(f1_score)

    print(precision_list)
    print(recall_list)
    print(f1_list)

    # マクロ平均を求める　f1は２種類の定義がある
    f1_mean = 0
    pre_mean = 0
    rec_mean = 0
    for i in range(4):
        f1_mean += f1_list[i]
        pre_mean += precision_list[i]
        rec_mean += recall_list[i]
    f1_mean = f1_mean/4
    pre_mean = pre_mean/4
    rec_mean = rec_mean/4
    f1_dash = (2*pre_mean*rec_mean)/(pre_mean+rec_mean)
    print(pre_mean)
    print(rec_mean)
    print(f1_mean)
    print(f1_dash)
    
    # マイクロ平均を求める
    TP = 0
    FP = 0
    for i in range(4):
        for j in range(4):
            if i == j:
                TP += con_mat[i][j]
            else:
                FP += con_mat[i][j]
    FN = FP
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = (2*pre*rec)/(pre+rec)
    print(pre)
    print(rec)
    print(f1)
    return 




# 正解のクラスと、特徴量を５３の関数に入れて予測クラスを得る
test_feature_df = pd.read_csv("test.feature.txt", sep='\t')
test_df = pd.read_csv("test.txt", sep='\t')
y_test = test_df["category"]
y_pred = p53.predict_category(test_feature_df)

# 55の関数で混同行列を得る
con_mat = p55.output_model_confusion_matrix(y_pred, y_test)
output_model_precision_recall_f1(con_mat)



'''
実行結果

カテゴリごと b, e, m, t
適合率　[0.8892215568862275, 0.9226415094339623, 0.6923076923076923, 0.5032679738562091]
再現率　[0.8342696629213483, 0.8939670932358318, 0.6666666666666666, 0.6936936936936937]
f１　[0.8608695652173913, 0.9080779944289694, 0.6792452830188679, 0.5833333333333334]

マクロ平均
適合率　0.7518596831210228
再現率　0.7721492791293851
f１　0.7578815439996405
f１　0.7618694203360683

マイクロ平均
適合率　0.8374429223744292
再現率　0.8374429223744292
f１　0.8374429223744292
'''