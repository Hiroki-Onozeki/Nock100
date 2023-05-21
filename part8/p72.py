import torch

# 単層のニューラルネットワークを作成して求める
class Single_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # wはランダムな初期値とする
        self.weights = torch.nn.Parameter(torch.randn(300, 4))
    
    def forward(self, x):
        y = x @ self.weights
        return y


def output_nn_ce_loss():
    train_features = torch.load("train_features.pt").float()
    train_labels = torch.load("train_labels.pt")

    x1 = train_features[:1]
    x2 = train_features[:4]
    y1 = train_labels[:1]
    y2 = train_labels[:4]

    # モデルに特徴量を入力し、予測確率を得る
    model = Single_nn()
    p1 = model(x1)
    p2 = model(x2)

    # クロスエントロピー損失を求める
    loss_func = torch.nn.CrossEntropyLoss()
    loss1 = loss_func(p1, y1)
    print(loss1)

    # 勾配を０に初期化してから逆転播し、勾配を出力する
    model.zero_grad()
    loss1.backward()
    print(model.weights.grad)

    # 同様に求める
    loss2 = loss_func(p2, y2)
    print(loss2)

    model.zero_grad()
    loss2.backward()
    print(model.weights.grad)
    return

output_nn_ce_loss()


'''
実行結果

tensor(1.5139, grad_fn=<NllLossBackward>)
tensor([[-1.8806e-03, -2.2397e-03,  4.5673e-03, -4.4701e-04],
        [-1.6137e-03, -1.9219e-03,  3.9192e-03, -3.8358e-04],
        [ 3.5291e-04,  4.2030e-04, -8.5709e-04,  8.3885e-05],
        ...,
        [-6.2160e-03, -7.4030e-03,  1.5097e-02, -1.4775e-03],
        [-1.0141e-03, -1.2077e-03,  2.4629e-03, -2.4105e-04],
        [-2.2622e-03, -2.6942e-03,  5.4941e-03, -5.3772e-04]])

tensor(1.3641, grad_fn=<NllLossBackward>)
tensor([[-2.0134e-04, -1.5363e-03,  1.6882e-03,  4.9427e-05],
        [ 1.6683e-03, -3.8202e-03,  2.0786e-03,  7.3230e-05],
        [-2.6858e-03, -2.0828e-03,  4.7844e-03, -1.5805e-05],
        ...,
        [-3.4178e-03, -7.5646e-03,  1.1445e-02, -4.6238e-04],
        [-4.4265e-07,  1.5988e-03, -1.5353e-03, -6.3126e-05],
        [-3.4109e-03,  2.7143e-03,  9.8819e-04, -2.9164e-04]])
'''