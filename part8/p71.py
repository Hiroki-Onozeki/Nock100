import torch


# 単層のニューラルネットワークを作成して求める
class Single_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(300, 4))
    
    def forward(self, x):
        softmax = torch.nn.Softmax(dim=1)
        y = x @ self.weights
        return softmax(y)


def output_nn_softmax():
    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")

    x1 = train_features[:1]
    x2 = train_features[:4]

    # モデルに入れて結果を得る
    model = Single_nn()
    y1 = model(x1)
    y2 = model(x2)

    print(y1)
    print(y2)
    print(train_labels[:4])
    return

output_nn_softmax()

'''
実行結果

tensor([[0.0535, 0.6696, 0.1275, 0.1494]], grad_fn=<SoftmaxBackward>)
tensor([[0.0535, 0.6696, 0.1275, 0.1494],
        [0.8942, 0.0081, 0.0181, 0.0797],
        [0.4850, 0.3859, 0.0407, 0.0885],
        [0.8244, 0.0315, 0.0159, 0.1282]], grad_fn=<SoftmaxBackward>)
tensor([3, 1, 0, 2])
'''