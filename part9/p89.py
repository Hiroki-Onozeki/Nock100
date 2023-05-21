import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import p80
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

class BERT(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = AutoModel.from_pretrained('bert-base-uncased')
    self.drop = torch.nn.Dropout(0.2)
    self.relu = torch.nn.ReLU()
    self.linear = torch.nn.Linear(768, 4)

  def forward(self, input_ids, attention_mask):
    last_hidden_state, pooler_output = self.bert(input_ids, attention_mask=attention_mask)
    drop_y = self.drop(pooler_output)
    relu_y = self.relu(drop_y) 
    y = self.linear(relu_y)
    return y



# モデルを学習させる
def train_model(b_size):
    train_dataset = torch.load("train_dataset_bert.pt")
    valid_dataset = torch.load("valid_dataset_bert.pt")
    epochs = 8
    batch_size = b_size
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    # モデル
    model = BERT()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_func = torch.nn.CrossEntropyLoss()

    # データローダーを作成
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()
        
        bar = tqdm(total=len(train_dataloader))
        for x_train, x_mask, y_train in train_dataloader:
            bar.update(1)
            x_train = x_train.to(device)
            x_mask = x_mask.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            pred = model(x_train, x_mask)
            loss = loss_func(pred, y_train)
            loss.backward()
            optimizer.step()

            # 損失と正解個数を保存
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
        for x_valid, v_mask, y_valid in valid_dataloader:
            x_valid = x_valid.to(device)
            v_mask = v_mask.to(device)
            y_valid = y_valid.to(device)
        
            pred = model(x_valid, v_mask)
            loss = loss_func(pred, y_valid)

            # 損失と正解個数を保存
            sum_loss += loss.item()
            pred_label = torch.argmax(pred, dim=1)
            acc_num += (pred_label == y_valid).sum().item()
            total += len(x_valid)
            loss_num += 1
        
    print(sum_loss/loss_num)
    print(acc_num/total)
    print(train_loss_list[-1])
    print(train_acc_list[-1])

    return


def make_data():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_labels = torch.load("train_labels.pt")
    valid_labels = torch.load("valid_labels.pt")
    train_df = pd.read_csv("train.txt", sep='\t')
    valid_df = pd.read_csv("valid.txt", sep='\t')
    train_title_df = train_df["title"]
    # 欠損値削除
    train_title_df = train_title_df.dropna()
    valid_title_df = valid_df["title"]
    train_input_ids = []
    train_attention_masks = []
    valid_input_ids =[]
    valid_attention_masks = []

    for title in train_title_df:
        words = title.split()
        encoded = tokenizer.encode_plus(
            words, # 文章
            padding = "max_length", # パディング方法
            max_length = 512, # 最大長さ
            truncation = True, # 最大長さを超えたら切り捨て
            return_tensors = "pt", 
            return_attention_mask = True,
            is_split_into_words=True
        )
        train_input_ids.append(encoded['input_ids'])
        train_attention_masks.append(encoded['attention_mask'])
    # リストに入ったtensorを縦方向（dim=0）へ結合
    train_input_ids = torch.cat(train_input_ids, dim=0)
    train_attention_masks = torch.cat(train_attention_masks, dim=0)

    for title in valid_title_df:
        words = title.split()
        encoded = tokenizer.encode_plus(
            words, # 文章
            padding = "max_length", # パディング方法
            max_length = 512, # 最大長さ
            truncation = True, # 最大長さを超えたら切り捨て
            return_tensors = "pt", 
            return_attention_mask = True,
            is_split_into_words=True
        )
        valid_input_ids.append(encoded['input_ids'])
        valid_attention_masks.append(encoded['attention_mask'])
    # リストに入ったtensorを縦方向（dim=0）へ結合
    valid_input_ids = torch.cat(valid_input_ids, dim=0)
    valid_attention_masks = torch.cat(valid_attention_masks, dim=0)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    valid_dataset = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)

    # ファイルに保存する
    torch.save(train_dataset, "train_dataset_bert.pt")
    torch.save(valid_dataset, "valid_dataset_bert.pt")
    return


# タイトルをトークナイザーに通した際の最大長を確認 1069
def count_title_length():
    train_df = pd.read_csv("train.txt", sep='\t')
    train_title_df = train_df["title"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    lengths = []

    for title in train_title_df:
        encoded = tokenizer.encode_plus(title.lower())
        lengths.append(len(encoded["input_ids"]))
    print(max(lengths))
    return

#count_title_length()
#make_data()
train_model(32)

"""
実行結果
ep4 normal
valid_loss  0.177966970562314
valid_acc  0.9482758620689655
train_loss  0.07964301334894792
train_acc   0.9767616191904048

ep4 dropout + relu
valid_loss  0.18664949524792887
valid_acc  0.9430284857571214
train_loss  0.11529019256373366
train_acc   0.9699212893553223

ep8 dropout + relu
valid_loss  0.22137197974093614
valid_acc  0.9430284857571214
train_loss  0.03860267362986764
train_acc   0.9922226386806596
"""