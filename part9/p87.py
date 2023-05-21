import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import part9.p80 as p80
from tqdm import tqdm


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, output_dim=50, label_num=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = torch.nn.Conv2d(1, output_dim, kernel_size=(3, 300), stride=1, padding=(1, 0))
        self.relu = torch.nn.ReLU()
        #self.pool = torch.MaxPool2d(kernel_size=())
        self.linear = torch.nn.Linear(output_dim, label_num)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        emb_x = self.embedding(x)
        # 次元追加
        emb_x = emb_x.unsqueeze(1)
        conv_x = self.conv(emb_x)
        relu_x = self.relu(conv_x)

        # 入力する単語数によって、poolingの引数が異なるため
        pool_x = torch.nn.functional.max_pool2d(relu_x, kernel_size=(relu_x.size()[2], 1))

        # 次元数を減らして整頓
        pool_x = torch.squeeze(pool_x, 2)
        pool_x = torch.squeeze(pool_x, 2)
        y = self.linear(pool_x)
        #y = self.softmax(y)
        return y
        


# モデルを学習させる
def train_model(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
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
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()

        bar = tqdm(total=len(train_dataloader))
        for x_train, y_train in train_dataloader:
            bar.update(1)
            # gpuに送る
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train)
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
            for x_valid, y_valid in valid_dataloader:
                # gpuに送る
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                pred = model(x_valid)
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
    

    print(train_loss_list)
    print(valid_loss_list)
    print(train_acc_list)
    print(valid_acc_list)
    return

word_id_dict = p80.return_word_id_dict()
vocab_size = len(word_id_dict) + 1 #pad
model = CNN(vocab_size=vocab_size, embedding_dim=300, output_dim=50, label_num=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(b_size=8)



"""
実行結果

train_loss  [20.087193478650182, 39.62307846774933, 12.831971699116297, 5.556225968984579, 3.067654905081613, 1.869249637695637, 1.3339293936765153, 0.895258567003073, 0.7267596182844546, 0.6440958758981595]
valid_loss  [26.58237885858886, 23.278534346557134, 13.23478293268468, 18.393743707151035, 19.460826032312962, 18.554986504054565, 17.305531865513462, 19.770862889347782, 20.18224789359662, 22.034604658658118]
train_acc   [0.5464767616191905, 0.6979010494752623, 0.8257121439280359, 0.8874625187406296, 0.9287856071964018, 0.9434032983508246, 0.9567091454272864, 0.966547976011994, 0.9706709145427287, 0.9736694152923538]
valid_acc   [0.6934032983508246, 0.7481259370314842, 0.7668665667166417, 0.782608695652174, 0.7968515742128935, 0.8215892053973014, 0.8223388305847077, 0.8095952023988006, 0.8193403298350824, 0.828335832083958]
"""