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


# モデルを学習させる
def train_nn_model():
    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")
    epochs = 100
    batch_size = 1

    # モデル、最適化手法、損失関数を指定する
    model = Single_nn()
    opt = torch.optim.SGD(model.parameters(), lr=0.1) #学習率
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        for x_train, y_train in train_dataloader:
            pred = model(x_train)
            loss = loss_func(pred, y_train)
            loss.backward()
            opt.step()  #パラメータの更新
            opt.zero_grad()  #勾配を０にする
    
    # モデルを保存
    torch.save(model.state_dict(), 'model_weights.pth')
    torch.save(model, 'model.pth')
    return


train_nn_model()
