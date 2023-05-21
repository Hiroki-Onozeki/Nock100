import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

# 単層のニューラルネットワークを作成して求める
class Single_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(300, 4))
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        y = x @ self.weights
        #y = self.softmax(y)
        return y


# モデルを学習させる
def train_model_with_batch(b_size):
    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt").float()
    valid_labels = torch.load("valid_labels.pt")
    epochs = 100
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # モデル、最適化手法、損失関数を指定する
    model = Single_nn()
    opt = torch.optim.SGD(model.parameters(), lr=0.1) #学習率
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()
        for x_train, y_train in train_dataloader:
            pred = model(x_train)
            loss = loss_func(pred, y_train)

            loss.backward()
            opt.step()  #パラメータの更新
            opt.zero_grad()  #勾配を０にする

            # 損失と正解個数を保存 b_size > 1 の場合は.item()が必要
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
                pred = model(x_valid)
                loss = loss_func(pred, y_valid)

                # 損失と正解個数を保存
                sum_loss += loss.sum().item()
                pred_label = torch.argmax(pred, dim=1)
                acc_num += (pred_label == y_valid).sum().item()
                total += len(x_valid)
                loss_num += 1
        
         # エポックごとに平均損失と正解率をリストに格納
        valid_loss_list.append(sum_loss/loss_num)
        valid_acc_list.append(acc_num/total)
        
    print(train_loss_list[-1], valid_loss_list[-1], train_acc_list[-1], valid_acc_list[-1])
    return

train_model_with_batch(b_size=1)

"""
実行結果

batch_size, time, train_loss, valid_loss, train_acc, valid_acc
1, 7:03.84 0.8287195402322356 0.8489133276860753 0.9252108716026242 0.8950524737631185
2, 3:12.50 0.8379142503148501 0.8474975965548491 0.9170571696344892 0.9010494752623688
4, 2:22.58 0.851672340257057 0.8535291871981706 0.9028116213683224 0.8995502248875562
8, 1:15.17 0.8663545159743108 0.8607875205085663 0.8889409559512652 0.8950524737631185
16, 0:44.73 0.8830584740710222 0.8775003545341038 0.8738519212746017 0.881559220389805
32, 0:29.95 0.9583488740250022 0.9468236650739398 0.7838800374882849 0.795352323838081
64, 0:20.02 0.9543098550356791 0.9516490470795405 0.7955951265229616 0.7991004497751124
128, 0:17.49 0.9806023750986371 0.9702392220497131 0.7691658856607311 0.7781109445277361
"""