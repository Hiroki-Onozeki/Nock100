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
def plot_model_loss_acc():
    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt").float()
    valid_labels = torch.load("valid_labels.pt")
    epochs = 100
    batch_size = 1
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
        model.train()
        for x_train, y_train in train_dataloader:
            pred = model(x_train)
            loss = loss_func(pred, y_train)

            # 損失と正解個数を保存
            sum_loss += loss
            pred_label = torch.argmax(pred)
            if(pred_label == y_train):
                acc_num += 1

            loss.backward()
            opt.step()  #パラメータの更新
            opt.zero_grad()  #勾配を０にする
        
        # エポックごとに平均損失と正解率をリストに格納
        train_loss_list.append(sum_loss/len(train_dataloader))
        train_acc_list.append(acc_num/len(train_dataloader))

        # 検証データ
        sum_loss = 0
        acc_num = 0
        model.eval()
        with torch.no_grad():#テンソルの計算結果を不可にしてメモリの消費を抑える
            for x_valid, y_valid in valid_dataloader:
                pred = model(x_valid)
                loss = loss_func(pred, y_valid)

                # 損失と正解個数を保存
                sum_loss += loss
                pred_label = torch.argmax(pred)
                if(pred_label == y_valid):
                    acc_num += 1
        
         # エポックごとに平均損失と正解率をリストに格納
        valid_loss_list.append(sum_loss/len(valid_dataloader))
        valid_acc_list.append(acc_num/len(valid_dataloader))

    # 折れ線グラフにプロットする
    epoch_num = list(range(1, 101))
    plt.plot(epoch_num, train_loss_list, label="train")
    plt.plot(epoch_num, valid_loss_list, label="valid")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("p75_loss.png")
    plt.figure()
    plt.plot(epoch_num, train_acc_list, label="train")
    plt.plot(epoch_num, valid_acc_list, label="valid")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig("p75_acc.png")
    return

plot_model_loss_acc()