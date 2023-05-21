import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import part9.p80 as p80
from tqdm import tqdm


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=50, label_num=4):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, label_num)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        # padding, packしてからlstmに送り、出力を解凍する
        x_len = torch.tensor([len(x1) for x1 in x])
        emb_x = self.embedding(x)
        packed_emb_x = pack_padded_sequence(emb_x, x_len, enforce_sorted=False, batch_first=True)
        # 各時刻の出力と、最後の隠れ層・セルの状態を出力する：最後の時系列のベクトルを使う　　h0,c0はデフォルトで０
        packed_output, hc = self.lstm(packed_emb_x)
        output, seq_len = pad_packed_sequence(packed_output, batch_first=True)
        y1 = self.linear(output[:,-1,:])
        y = self.softmax(y1)
        return y

# モデルを学習させる
def output_y():
    train_features = torch.load("train_features.pt")
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict)


    # モデル、最適化手法、損失関数を指定する
    model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=50, label_num=4)
    x_train = train_features[1:2]
    y = model(x_train)
    print(y)
    return


# ファイル作成
def make_feature_label_file():
    train_df = pd.read_csv("train.txt", sep='\t')
    test_df = pd.read_csv("test.txt", sep='\t')
    valid_df = pd.read_csv("valid.txt", sep='\t')

    # タイトルを抽出し、特徴量に変換する
    train_title_df = train_df["title"]
    test_title_df = test_df["title"]
    valid_title_df = valid_df["title"]
    train_title_vector = p80.return_word_id(train_title_df)
    test_title_vector = p80.return_word_id(test_title_df)
    valid_title_vector = p80.return_word_id(valid_title_df)

    # 長さを保存する
    train_len = torch.tensor([len(x) for x in train_title_vector])
    test_len = torch.tensor([len(x) for x in test_title_vector])
    valid_len = torch.tensor([len(x) for x in valid_title_vector])

    # 長さを揃える -> tensorに変換するため
    train_title_pt = [torch.tensor(x) for x in train_title_vector]
    test_title_pt = [torch.tensor(x) for x in test_title_vector]
    valid_title_pt = [torch.tensor(x) for x in valid_title_vector]

    padded_train_title_pt = pad_sequence(train_title_pt, batch_first=True)
    padded_test_title_pt = pad_sequence(test_title_pt, batch_first=True)
    padded_valid_title_pt = pad_sequence(valid_title_pt, batch_first=True)

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
    torch.save(padded_train_title_pt, "train_features.pt")
    torch.save(padded_test_title_pt, "test_features.pt")
    torch.save(padded_valid_title_pt, "valid_features.pt")
    torch.save(train_labels_pt, "train_labels.pt")
    torch.save(test_labels_pt, "test_labels.pt")
    torch.save(valid_labels_pt, "valid_labels.pt")
    torch.save(train_len, "train_len.pt")
    torch.save(test_len, "test_len.pt")
    torch.save(valid_len, "valid_len.pt")
    return

output_y()
#make_feature_label_file()

"""
実行結果

tensor([[0.2351, 0.2471, 0.2493, 0.2685]], grad_fn=<SoftmaxBackward0>)

b    4499
e    4193
t    1241
m     739
"""