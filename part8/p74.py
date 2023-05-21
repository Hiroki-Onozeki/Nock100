import torch
from torch.utils.data import TensorDataset, DataLoader

# 単層のニューラルネットワークを作成して求める
class Single_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(300, 4))
    
    def forward(self, x):
        y = x @ self.weights
        return y

# 正解率を出力する
def output_accuracy():
    sum_num = 0
    acc_num = 0
    model = torch.load('model.pth')
    model.eval()

    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")
    test_features = torch.load("test_features.pt").float()
    test_labels = torch.load("test_labels.pt")

    # データローダーを作成
    batch_size = 1
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(test_features, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    with torch.no_grad():
        for x_train, y_train in train_dataloader:
            sum_num += 1
            pred_prob = model(x_train)
            pred_label = torch.argmax(pred_prob)
            if(pred_label == y_train):
                acc_num += 1
    train_acc = acc_num / sum_num
    sum_num = 0
    acc_num = 0

    # 評価データをモデルに入力し、予測確率が最大となるラベル(=インデックス)と正解ラベルを比べる
    with torch.no_grad():#テンソルの計算結果を不可にしてメモリの消費を抑える
        for x_test, y_test in test_dataloader:
            sum_num += 1
            pred_prob = model(x_test)
            pred_label = torch.argmax(pred_prob)
            if(pred_label == y_test):
                acc_num += 1
    test_acc = acc_num / sum_num
    return train_acc, test_acc

print(output_accuracy())
            

"""
実行結果

train   0.9271790065604498
test    0.8913043478260869
"""