import gensim
import pandas as pd
import torch
import numpy as np


def make_feature_label_file():
    train_df = pd.read_csv("train.txt", sep='\t')
    test_df = pd.read_csv("test.txt", sep='\t')
    valid_df = pd.read_csv("valid.txt", sep='\t')

    # タイトルを抽出し、特徴量に変換する
    train_title_df = train_df["title"]
    test_title_df = test_df["title"]
    valid_title_df = valid_df["title"]
    train_title_vector = output_title_vector(train_title_df)
    test_title_vector = output_title_vector(test_title_df)
    valid_title_vector = output_title_vector(valid_title_df)
    train_title_pt = torch.from_numpy(np.array(train_title_vector))
    test_title_pt = torch.from_numpy(np.array(test_title_vector))
    valid_title_pt = torch.from_numpy(np.array(valid_title_vector))

    # ラベルを抽出し、置き換える
    train_labels_df = train_df["category"]
    train_labels_df = train_labels_df.replace(['b','t','e','m'], [0,1,2,3])
    test_labels_df = test_df["category"]
    test_labels_df = test_labels_df.replace(['b','t','e','m'], [0,1,2,3])
    valid_labels_df = valid_df["category"]
    valid_labels_df = valid_labels_df.replace(['b','t','e','m'], [0,1,2,3])

    # テンソルに変換する
    train_labels_pt = torch.tensor(train_labels_df)
    test_labels_pt = torch.tensor(test_labels_df)
    valid_labels_pt = torch.tensor(valid_labels_df)

    # ファイルに保存する
    torch.save(train_title_pt, "train_features.pt")
    torch.save(test_title_pt, "test_features.pt")
    torch.save(valid_title_pt, "valid_features.pt")
    torch.save(train_labels_pt, "train_labels.pt")
    torch.save(test_labels_pt, "test_labels.pt")
    torch.save(valid_labels_pt, "valid_labels.pt")
    return

def output_title_vector(title_df):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    all_title_vector = []
    index = -1

    for title in title_df:
        sum_title_vector = []
        title_vector = []
        word_num = 0
        word_list = title.split()
        index += 1

        # 各単語のベクトルを計算し、二次元リストへ
        for word in word_list:
            if word in model:
                vector = model[word]
                word_num += 1
                title_vector.append(vector)

        # trainに空のやつがあるらしい、それを弾く
        if title_vector == []:
            print(index)
            continue
        
        # 二次元リストから各行の値を加算した配列を求める
        for i in range(len(title_vector[0])):
            sum_elem = 0
            for j in range(len(title_vector)):
                sum_elem += title_vector[j][i]
            sum_title_vector.append(sum_elem/word_num)
        
        # 全体のリストに格納し、返す
        all_title_vector.append(sum_title_vector) 
    return all_title_vector

""" 
make_feature_label_file()

train_labels = torch.load("train_labels.pt")
train_labels_np = train_labels.numpy()
train_delete = np.delete(train_labels_np, [220, 371])
train_labels_pt = torch.from_numpy(train_delete)
torch.save(train_labels_pt, "train_labels.pt")
"""

