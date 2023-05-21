import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import p80
from tqdm import tqdm
import optuna
import gensim


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, label_num=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = torch.nn.Conv2d(1, output_dim, kernel_size=(3, embedding_dim), stride=1, padding=(1, 0))
        self.relu = torch.nn.ReLU()
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
        
class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, label_num, weights, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(weights)
        #self.embedding = torch.nn.Embedding.from_pretrained(weights)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim*2, label_num)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x, x_len):
        # padding, packしてからlstmに送り、出力を解凍する
        x_len = x_len.cpu()
        emb_x = self.embedding(x)
        packed_emb_x = pack_padded_sequence(emb_x, x_len, enforce_sorted=False, batch_first=True)

        # 出力が双方向分２倍になる、ht,h1を使う   h0,c0はデフォルトで０
        packed_output, hc = self.lstm(packed_emb_x)
        h0t = torch.cat([hc[0][0], hc[0][1]], dim=1)
        y = self.linear(h0t)
        #y = self.softmax(y)
        return y


# モデルを学習させる
def train_model_cnn(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
    epochs = 200
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    model = CNN(vocab_size=vocab_size, embedding_dim=400, output_dim=30, label_num=4)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=0.001) #学習率
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    bar = tqdm(total=epochs)
    for epoch in range(epochs):
        bar.update(1)
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()

        for x_train, y_train in train_dataloader:
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
        print(sum_loss/loss_num)
        print(acc_num/total)
    

    print(train_loss_list[-1])
    print(train_acc_list[-1])
    return


# モデルを学習させる
def train_model_lstm(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
    train_len = torch.load("train_len.pt").int()
    valid_len = torch.load("valid_len.pt").int()
    epochs = 80
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    weights = np.zeros((vocab_size, 300))
    for idx, elem in enumerate(word_id_dict.keys()):  
        if elem in wv_model:
            weights[idx] = wv_model[elem]
        else:
            weights[idx] = np.random.randn(300)
    weights = torch.from_numpy(np.array(weights).astype((np.float32)))

    model = LSTM(vocab_size=vocab_size, hidden_dim=70, embedding_dim=300, label_num=4, weights=weights, num_layers=4)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=0.0001) #学習率
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels, train_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels, valid_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    bar = tqdm(total=epochs)
    for epoch in range(epochs):
        bar.update(1)
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()

        for x_train, y_train, x_len in train_dataloader:
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
    print(train_loss_list[-1])
    print(valid_loss_list[-1])
    print(train_acc_list[-1])
    print(valid_acc_list[-1])
    return



def objective(trial):
    # ハイパーパラメータの設定
    embedding_dim = int(trial.suggest_discrete_uniform('embedding_dim', 100, 500, 100))
    output_dim = int(trial.suggest_discrete_uniform('output_dim', 10, 100, 10))
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 8, 64, 8))
    epochs = int(trial.suggest_discrete_uniform('epochs', 50, 200, 10))
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    train_features = torch.load("train_features.pt")
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")
    valid_labels = torch.load("valid_labels.pt")
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    model = CNN(vocab_size=vocab_size, embedding_dim=embedding_dim, output_dim=output_dim, label_num=4)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x_train, y_train in train_dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train)
            loss = loss_func(pred, y_train)
            loss.backward()
            opt.step()  
            opt.zero_grad()  

    # 検証データ
    sum_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x_valid, y_valid in valid_dataloader:
            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)
            pred = model(x_valid)
            loss = loss_func(pred, y_valid)
            sum_loss += loss.item()
            total += 1
        loss = sum_loss/total
    return loss

def objective_lstm(trial):
    # ハイパーパラメータの設定
    embedding_dim = int(trial.suggest_discrete_uniform('embedding_dim', 100, 500, 100))
    hidden_dim = int(trial.suggest_discrete_uniform('output_dim', 10, 100, 10))
    batch_size = int(trial.suggest_discrete_uniform('batch_size', 8, 64, 8))
    epochs = int(trial.suggest_discrete_uniform('epochs', 50, 200, 10))
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    num_layers = int(trial.suggest_discrete_uniform('num_layers', 1, 8, 1))

    train_features = torch.load("train_features.pt")
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")
    valid_labels = torch.load("valid_labels.pt")
    train_len = torch.load("train_len.pt").int()
    valid_len = torch.load("valid_len.pt").int()
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    weights = np.zeros((vocab_size, 300))
    for idx, elem in enumerate(word_id_dict.keys()):  
        if elem in wv_model:
            weights[idx] = wv_model[elem]
        else:
            weights[idx] = np.random.randn(300)
    weights = torch.from_numpy(np.array(weights).astype((np.float32)))

    model = LSTM(vocab_size=vocab_size, hidden_dim=hidden_dim, embedding_dim=embedding_dim, label_num=4, weights=weights, num_layers=num_layers)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels, train_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels, valid_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x_train, y_train, x_len in train_dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train, x_len)
            loss = loss_func(pred, y_train)
            loss.backward()
            opt.step()  
            opt.zero_grad()  

    # 検証データ
    sum_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x_valid, y_valid, x_len in valid_dataloader:
            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)
            pred = model(x_valid, x_len)
            loss = loss_func(pred, y_valid)
            sum_loss += loss.item()
            total += 1
        loss = sum_loss/total
    return loss

def output_optimal_param():
    # objectiveの出力がlossなので最小化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, timeout=600)

    # ベストパラメータを出力
    #print(study.best_params)
    #print(study.best_value)
    print(study.best_trial)
    return


train_model_cnn(b_size=8)
#train_model_lstm(b_size=1)
#output_optimal_param()


"""
lstm
Trial 0 finished with value: 1.1541659924246015 and parameters: {'embedding_dim': 300.0, 'output_dim': 70.0, 'batch_size': 16.0, 'epochs': 80.0, 'learning_rate': 0.0001, 'num_layers': 4.0}.
valid_loss  1.0473385030123783
valid_acc   0.8545727136431784
train_loss  0.001939161826338614
train_acc   0.9989692653673163

cnn
Trial 1 finished with value: 0.520106703042984 and parameters: {'embedding_dim': 400.0, 'output_dim': 30.0, 'batch_size': 64.0, 'epochs': 200.0, 'learning_rate': 0.001}. Best is trial 0 with value: 0.520106703042984.
valid_loss  0.5105360320636204
valid_acc   0.8313343328335832
train_loss  0.004973531563594732
train_acc   0.9988755622188905
"""