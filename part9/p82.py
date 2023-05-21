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
        # padding, packしてからlstmに送り、出力を解凍する, x_lenはcpuに
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
def train_model():
    train_features = torch.load("train_features.pt")
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")
    valid_labels = torch.load("valid_labels.pt")
    train_len = torch.load("train_len.pt").int()
    valid_len = torch.load("valid_len.pt").int()
    epochs = 10
    batch_size = 1
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
            total += 1
            sum_loss += loss.item()
            pred_label = torch.argmax(pred)
            if(pred_label == y_train):
                acc_num += 1
        
        # エポックごとに平均損失と正解率をリストに格納
        train_loss_list.append(sum_loss/total)
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
                total += 1
                sum_loss += loss.item()
                pred_label = torch.argmax(pred)
                if(pred_label == y_valid):
                    acc_num += 1
        
         # エポックごとに平均損失と正解率をリストに格納
        valid_loss_list.append(sum_loss/total)
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
train_model()


"""
実行結果

train_loss  [0.8921517821310644, 0.6125085682619642, 0.43777872346541225, 0.32801137404596414, 0.276573222839662, 0.20575809976202653, 0.14990103971312646, 0.1462592688913955, 0.12359473051422645, 0.09183603234709817]
valid_loss  [0.7271254702827992, 0.6840711587311067, 0.6883208343670546, 0.7259922371309585, 0.7795394322671965, 0.828843137941397, 1.0067713653290307, 0.996202837259395, 1.0351549486702207, 1.0533436495054418]
train_acc   [0.6741004497751124, 0.7801724137931034, 0.8479197901049476, 0.8860569715142429, 0.9015179910044977, 0.9311281859070465, 0.9469640179910045, 0.9485569715142429, 0.9583958020989505, 0.9679535232383808]
valid_acc   [0.7556221889055472, 0.767616191904048, 0.7713643178410795, 0.7683658170914542, 0.7743628185907047, 0.7773613193403298, 0.7631184407796102, 0.7908545727136432, 0.7998500749625187, 0.7908545727136432]
"""