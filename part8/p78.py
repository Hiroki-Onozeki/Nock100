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
    #model = Single_nn()
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
        
    print(train_loss_list[-1], valid_loss_list[-1], train_acc_list[-1], valid_acc_list[-1])
    return

model = Single_nn()
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model_with_batch(b_size=4)

"""
1, 20:23.72 0.8294121003586663 0.8500676587156984 0.9254920337394564 0.8913043478260869
2, 19:40.62 0.8383620779650616 0.8484585558635362 0.9154639175257732 0.896551724137931
4, 13:13.09 0.8486101660860711 0.8529360885034778 0.9049671977507029 0.8913043478260869
8, 6:48.45 0.11239274650281461 0.11214544913400595 0.8512652296157451 0.8583208395802099
16, 3:37.12 0.9327139865154627 0.9246761067992165 0.8158388003748829 0.8260869565217391
32, 2:00.68 0.9361149126184201 0.93204389441581 0.8149015932521088 0.8185907046476761
64, 1:10.22 0.96852072662936 0.9543911332175845 0.777788191190253 0.7968515742128935
128, 0:45.68 0.9739555446874528 0.9653544100848112 0.7749765698219306 0.7841079460269865
"""