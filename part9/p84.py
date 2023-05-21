import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import part9.p80 as p80
from tqdm import tqdm
import gensim


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, label_num, weights):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(weights)
        #self.embedding = torch.nn.Embedding.from_pretrained(weights)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, label_num)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x, x_len):
        # padding, packしてからlstmに送り、出力を解凍する
        x_len = x_len.cpu()
        emb_x = self.embedding(x)
        packed_emb_x = pack_padded_sequence(emb_x, x_len, enforce_sorted=False, batch_first=True)
        # 各時刻の出力と、最後の隠れ層・セルの状態を出力する：最後の時系列のベクトルを使う   h0,c0はデフォルトで０
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



# 重みを設定：out of memoryになるので置き換える
wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word_id_dict = p80.return_word_id_dict()
vocab_size = len(word_id_dict) + 1
weights = np.zeros((vocab_size, 300))
for idx, elem in enumerate(word_id_dict.keys()):  
    if elem in wv_model:
        weights[idx] = wv_model[elem]
    else:
        weights[idx] = np.random.randn(300)
weights = torch.from_numpy(np.array(weights).astype((np.float32)))

model = LSTM(vocab_size=vocab_size, hidden_dim=50, embedding_dim=300, label_num=4, weights=weights)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(b_size=8)


"""
実行結果

b=8
[0.8334133304666722, 0.6899085419288818, 0.6312236416040511, 0.5883798323839768, 0.5569423800376223, 0.5290347358164401, 0.513580367963264, 0.497615748404831, 0.4827939577769304, 0.47483643917352897]
[0.7162241084489994, 0.6565544059176645, 0.6432908702039433, 0.6268790964773315, 0.617684481386653, 0.5972200421516053, 0.5806852141004837, 0.5890669650481847, 0.5918808943497207, 0.5948994805712899]
[0.5490067466266867, 0.6231259370314842, 0.6393365817091454, 0.6560157421289355, 0.6640742128935532, 0.6711956521739131, 0.6785982008995503, 0.6891866566716641, 0.6966829085457271, 0.7031484257871065]
[0.618440779610195, 0.6416791604197901, 0.643928035982009, 0.6551724137931034, 0.658920539730135, 0.664167916041979, 0.6671664167916042, 0.6709145427286357, 0.671664167916042, 0.6836581709145427]

b=1
[0.8204529085713399, 0.42746784158300166, 0.18522639511393846, 0.06281915154437187, 0.028428755329740627, 0.011374201153562144, 0.0051911565991745005, 0.004572174680486103, 0.0034526683048068593, 0.0033090925854399266]
[0.6168889139912955, 0.4326653909550626, 0.4510567690823532, 0.5091319409089211, 0.5774519810774735, 0.6921221308983164, 0.7255322319305615, 0.750977668601086, 0.7574094707668251, 0.7830014112333389]
[0.697151424287856, 0.8407983508245878, 0.9360007496251874, 0.9819152923538231, 0.9913793103448276, 0.9959707646176912, 0.99821964017991, 0.99821964017991, 0.9987818590704648, 0.998688155922039]
[0.7781109445277361, 0.8523238380809596, 0.8598200899550225, 0.8680659670164917, 0.8658170914542729, 0.8703148425787106, 0.8755622188905547, 0.8770614692653673, 0.8725637181409296, 0.8755622188905547]
"""