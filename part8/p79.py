import torch
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

# ニューラルネットワーク
class Multi_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.l1 = torch.nn.Linear(300, 32)
        torch.nn.init.constant_(self.l1.weight, 0)
        torch.nn.init.constant_(self.l1.bias, 0)
        self.b1 = torch.nn.BatchNorm1d(32)
        self.l2 = torch.nn.Linear(32, 4)
        torch.nn.init.constant_(self.l2.weight, 0)
        torch.nn.init.constant_(self.l2.bias, 0)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.b1(x)
        x = self.l2(x)
        return x

class Single_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(300, 4))
        # 初期値を０に固定、初期値による結果のバラツキが大きい?
        torch.nn.init.constant_(self.weights, 0)
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
    epochs = 1000
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # モデル、最適化手法、損失関数を指定する
    model = Single_nn()
    #model = Multi_nn()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3) #学習率
    loss_func = torch.nn.CrossEntropyLoss()
    bar = tqdm(total=epochs)

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
        for x_train, y_train in train_dataloader:

            # gpuに送る
            x_train = x_train.to(device)
            y_train = y_train.to(device)

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
        train_loss_list.append(sum_loss/total)
        train_acc_list.append(acc_num/total)

        # 検証データ
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.eval()
        with torch.no_grad():#テンソルの計算結果を不可にしてメモリの消費を抑える
            for x_valid, y_valid in valid_dataloader:
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
        valid_loss_list.append(sum_loss/total)
        valid_acc_list.append(acc_num/total)
        bar.update(1)

        if epoch > 100:
            if valid_loss_list[epoch - 3] <= valid_loss_list[epoch - 2] <= valid_loss_list[epoch - 1] <= valid_loss_list[epoch]:
                print(epoch)
                break
        
    print(train_acc_list[-1], valid_acc_list[-1])
    return


train_model_with_batch(b_size=128)



"""
実行結果

128 b_size
Single  0.7720712277413309 0.7833583208395802
2層     0.5094657919400187 0.5097451274362819
4層     0.4216494845360825 0.42278860569715143

1 b_size
Single   0.9020618556701031 0.8958020989505248
2層      0.9034676663542643 0.8950524737631185
"""

