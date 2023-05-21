import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import part9.p80 as p80
from tqdm import tqdm


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, output_dim=50, label_num=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = torch.nn.Conv2d(1, output_dim, kernel_size=(3, 300), stride=1, padding=(1, 0))
        self.relu = torch.nn.ReLU()
        #self.pool = torch.MaxPool2d(kernel_size=())
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
        y = self.softmax(y)
        return y
"""
emb_x torch.Size([1, 1, 201, 300])
conv_x torch.Size([1, 50, 201, 1])
relu_x torch.Size([1, 50, 201, 1])
pool_x torch.Size([1, 50, 1, 1])
pool_x torch.Size([1, 50])
"""


# モデルを学習させる
def train_model(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
    epochs = 10
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    word_id_dict = p80.return_word_id_dict()
    vocab_size = len(word_id_dict) + 1 #pad

    # モデル、最適化手法、損失関数を指定する
    opt = torch.optim.SGD(model.parameters(), lr=0.1) #学習率
    weights = torch.tensor([0.42, 0.12, 0.39, 0.07]).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=weights)
    #loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    
    for x_train, y_train in train_dataloader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        pred = model(x_train)
        print(pred)
        break 
        
    return


word_id_dict = p80.return_word_id_dict()
vocab_size = len(word_id_dict) + 1 #pad
model = CNN(vocab_size=vocab_size, embedding_dim=300, output_dim=50, label_num=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(b_size=1)


"""
実行結果

tensor([[0.2227, 0.1107, 0.5670, 0.0996]], device='cuda:0',
       grad_fn=<GatherBackward>)
"""