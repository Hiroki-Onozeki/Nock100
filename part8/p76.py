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
def output_checkpoint_each_ep():
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
            opt.zero_grad()
            pred = model(x_train)
            loss = loss_func(pred, y_train)

            loss.backward()
            opt.step()  #パラメータの更新

        # ファイルにモデルと最適化アルゴリズムのパラメータを保存
        checkpoint = {"model": model.state_dict(), "optimizer":opt.state_dict()}
        print(checkpoint)
        #torch.save(checkpoint, "p76_checkpoints/checkpoint_" + str(epoch) + ".pt")

    return

output_checkpoint_each_ep()

"""
実行結果

{'model': OrderedDict([('weights', tensor([[-0.5368,  0.0476, -0.6168, -0.4398],
        [-0.6159, -0.1977, -0.8622, -0.2117],
        [ 0.0288, -0.3998, -0.3162, -0.1573],
        ...,
        [ 1.4083, -0.7356,  0.3908, -1.0836],
        [ 0.7233,  0.9197,  1.5671, -0.9765],
        [-0.5573, -0.8218,  0.0545, -1.8312]]))]), 
'optimizer': {'state': {}, 
'param_groups': [{'lr': 0.1, 
'momentum': 0, 'dampening': 0, 
'weight_decay': 0, 
'nesterov': False, 
'params': [140352914550528]}]}}
"""