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
    
    def forward(self, x, x_len):
        # padding, packしてからlstmに送り、出力を解凍する
        x_len = x_len.cpu()
        emb_x = self.embedding(x)
        packed_emb_x = pack_padded_sequence(emb_x, x_len, enforce_sorted=False, batch_first=True)
        # 各時刻の出力と、最後の隠れ層・セルの状態を出力する：最後の時系列のベクトルを使う  h0,c0はデフォルトで０
        packed_output, hc = self.lstm(packed_emb_x)
        output, seq_len = pad_packed_sequence(packed_output, batch_first=True)
        y = self.linear(output[:,-1,:])
        #y = self.softmax(y)
        return y



# モデルを学習させる
def train_model(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
    train_len = torch.load("train_len.pt").int()
    valid_len = torch.load("valid_len.pt").int()
    epochs = 10
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=0.1) #学習率
    weights = torch.tensor([0.42, 0.12, 0.39, 0.07]).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    #loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels, train_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels, valid_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()

        bar = tqdm(total=len(train_dataloader))
        for x_train, y_train, x_len in train_dataloader:
            bar.update(1)
            # gpuに送る
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train, x_len)
            loss = loss_func(pred, y_train)
            loss.backward()
            opt.step()  #パラメータの更新
            opt.zero_grad()  #勾配を０にする

            # 損失と正解個数を保存
            sum_loss += loss.item()
            pred_label = torch.argmax(pred, dim=1)
            acc_num += (pred_label == y_train).sum().item()
            total += len(x_train)
            loss_num += 1
        
        # エポックごとに平均損失と正解率をリストに格納
        train_loss_list.append(sum_loss/loss_num)
        train_acc_list.append(acc_num/total)

        # 検証データ
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.eval()
        with torch.no_grad():#テンソルの計算結果を不可にしてメモリの消費を抑える
            for x_valid, y_valid, x_len in valid_dataloader:
                # gpuに送る
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                pred = model(x_valid, x_len)
                loss = loss_func(pred, y_valid)

                # 損失と正解個数を保存
                sum_loss += loss.item()
                pred_label = torch.argmax(pred, dim=1)
                acc_num += (pred_label == y_valid).sum().item()
                total += len(x_valid)
                loss_num += 1
        
         # エポックごとに平均損失と正解率をリストに格納
        valid_loss_list.append(sum_loss/loss_num)
        valid_acc_list.append(acc_num/total)
    
    print("-----result-----")
    print(train_loss_list)
    print(valid_loss_list)
    print(train_acc_list)
    print(valid_acc_list)
    return

word_id_dict = p80.return_word_id_dict()
vocab_size = len(word_id_dict) + 1 #pad
model = LSTM(vocab_size=vocab_size, embedding_dim=300, hidden_dim=50, label_num=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(b_size=8)


"""
実行結果

train_loss  [1.0734330869015904, 0.924219116166435, 0.8103264056626467, 0.7369282386940101, 0.6753577154012456, 0.6218256896425938, 0.5949692655539048, 0.5801401445600225, 0.5688197322446724, 0.5408154145676246]
valid_loss  [0.9700989858832901, 0.9154040408348609, 0.8820122216276066, 0.855062467609337, 0.8376246130395079, 0.9233810868627297, 0.9183889297311177, 0.9483061169614335, 0.9629668235600352, 0.9639390676678298]
train_acc   [0.5592203898050975, 0.6304347826086957, 0.6708208395802099, 0.6941529235382309, 0.7223575712143928, 0.7427848575712144, 0.7560907046476761, 0.7568403298350824, 0.761900299850075, 0.7713643178410795]
valid_acc   [0.6191904047976012, 0.6499250374812594, 0.656671664167916, 0.6821589205397302, 0.6776611694152923, 0.6626686656671664, 0.6724137931034483, 0.6634182908545727, 0.6709145427286357, 0.6776611694152923]
"""
