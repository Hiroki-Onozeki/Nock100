from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import os
import numpy as np

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        # 次元や転置を戻す
        src_mask = src_mask[0]
        tgt_mask = tgt_mask[0]
        src = torch.t(src)
        trg = torch.t(trg)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask



def train_epoch(model, optimizer, dataloader):
    model.train()
    losses = 0

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # batch firstを戻す
        src = torch.t(src)
        tgt = torch.t(tgt)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)

        # パラレルのために次元追加や転置をする
        src = torch.t(src)
        tgt_input = torch.t(tgt_input)
        src_mask = src_mask.repeat(src.shape[0], 1, 1)
        tgt_mask = tgt_mask.repeat(tgt.shape[0], 1, 1)     
        #print(src.size())
        #print(tgt_input.size())

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()
        logits = logits.transpose(0, 1)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        if math.isnan(loss) == False:
            loss.backward()
            losses += loss.item()
            optimizer.step()
 
    return losses / len(dataloader)


def evaluate(model, dataloader):
    model.eval()
    losses = 0

    for src, tgt in dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src = torch.t(src)
        tgt = torch.t(tgt)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)


        # パラレルのために次元追加や転置をする
        src = torch.t(src)
        tgt_input = torch.t(tgt_input)
        src_mask = src_mask.repeat(src.shape[0], 1, 1)
        tgt_mask = tgt_mask.repeat(tgt.shape[0], 1, 1)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        logits = logits.transpose(0, 1)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        if math.isnan(loss) == False:
            losses += loss.item()

    return losses / len(dataloader)



def train_model(output_file_name1, output_file_name2, train_dataloader, test_dataloader):
    print(str(NUM_EPOCHS) + "epochs")
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()
        val_loss = evaluate(transformer, test_dataloader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer.state_dict(), output_file_name1)
    torch.save(transformer, output_file_name2)
    print("---done---")
    return

def load_data(batch_size):
    train_data_ja = torch.load("train_data/s_train_data_ja.pt")
    test_data_ja = torch.load("train_data/s_test_data_ja.pt")
    train_data_en = torch.load("train_data/s_train_data_en.pt")
    test_data_en = torch.load("train_data/s_test_data_en.pt")

    #train_data_ja = torch.load("train_data/s_dev_data_ja.pt")
    #train_data_en = torch.load("train_data/s_dev_data_en.pt")
    #test_data_ja = torch.load("train_data/s_dev_test_data_ja.pt")
    #test_data_en = torch.load("train_data/s_dev_test_data_en.pt")

    train_dataset = TensorDataset(train_data_ja, train_data_en)
    test_dataset = TensorDataset(test_data_ja, test_data_en)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader

BATCH_SIZE = 64
NUM_EPOCHS = 30
PAD_IDX = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader, test_dataloader = load_data(BATCH_SIZE)

# 辞書を読み込む
ja_dict = np.load("ja_dict.npy", allow_pickle=True)
ja_dict = ja_dict.item()
en_dict = np.load("en_dict.npy", allow_pickle=True)
en_dict = en_dict.item()
SRC_VOCAB_SIZE = len(ja_dict) + 4
TGT_VOCAB_SIZE = len(en_dict) + 4

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
torch.manual_seed(0)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

transformer = torch.nn.DataParallel(transformer, device_ids=[0, 1, 2, 3])
transformer = transformer.to(DEVICE)


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

train_model("model/dev_model_weight_30ep.pth", "model/dev_model_30ep.pth", train_dataloader, test_dataloader)

#nohup python3 -u p91-orig.py > & output.txt &

"""
実行結果
Epoch: 1 Train loss: 4.6572, Valid loss: 3.7600, Epoch time: 968.3909
Epoch: 2 Train loss: 3.4701, Valid loss: 3.0048, Epoch time: 949.0514
Epoch: 3 Train loss: 2.9913, Valid loss: 2.6796, Epoch time: 942.8674
Epoch: 4 Train loss: 2.7217, Valid loss: 2.4946, Epoch time: 935.0439
Epoch: 5 Train loss: 2.5437, Valid loss: 2.3991, Epoch time: 946.2389
Epoch: 6 Train loss: 2.4166, Valid loss: 2.2995, Epoch time: 945.5730
Epoch: 7 Train loss: 2.3201, Valid loss: 2.2537, Epoch time: 938.0471
Epoch: 8 Train loss: 2.2432, Valid loss: 2.2075, Epoch time: 948.6539
Epoch: 9 Train loss: 2.1799, Valid loss: 2.1794, Epoch time: 949.9027
Epoch: 10 Train loss: 2.1262, Valid loss: 2.1383, Epoch time: 949.1726
Epoch: 11 Train loss: 2.0810, Valid loss: 2.1376, Epoch time: 950.5201
Epoch: 12 Train loss: 2.0407, Valid loss: 2.1227, Epoch time: 950.5228
Epoch: 13 Train loss: 2.0058, Valid loss: 2.1136, Epoch time: 946.8743
Epoch: 14 Train loss: 1.9742, Valid loss: 2.0890, Epoch time: 947.8143
Epoch: 15 Train loss: 1.9460, Valid loss: 2.0843, Epoch time: 950.4059
Epoch: 16 Train loss: 1.9205, Valid loss: 2.0899, Epoch time: 946.2043
Epoch: 17 Train loss: 1.8968, Valid loss: 2.0903, Epoch time: 944.4603
Epoch: 18 Train loss: 1.8752, Valid loss: 2.0743, Epoch time: 954.2637
Epoch: 19 Train loss: 1.8551, Valid loss: 2.0645, Epoch time: 957.2912
Epoch: 20 Train loss: 1.8371, Valid loss: 2.0561, Epoch time: 953.1029
Epoch: 21 Train loss: 1.8199, Valid loss: 2.0690, Epoch time: 953.5907
Epoch: 22 Train loss: 1.8044, Valid loss: 2.0608, Epoch time: 948.6677
Epoch: 23 Train loss: 1.7891, Valid loss: 2.0586, Epoch time: 947.7270
Epoch: 24 Train loss: 1.7753, Valid loss: 2.0668, Epoch time: 944.7827
Epoch: 25 Train loss: 1.7618, Valid loss: 2.0657, Epoch time: 939.1242
Epoch: 26 Train loss: 1.7492, Valid loss: 2.0612, Epoch time: 933.9640
Epoch: 27 Train loss: 1.7372, Valid loss: 2.0639, Epoch time: 946.1230
Epoch: 28 Train loss: 1.7266, Valid loss: 2.0483, Epoch time: 956.1468
Epoch: 29 Train loss: 1.7163, Valid loss: 2.0583, Epoch time: 931.3928
Epoch: 30 Train loss: 1.7063, Valid loss: 2.0572, Epoch time: 932.3962
"""

