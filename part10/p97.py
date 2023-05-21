from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import sentencepiece as spm


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
        #print("src" + str(src.size()))
        #print("trg" +str(trg.size()))

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


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, dataloader):
    model.train()
    losses = 0

    for src, tgt in dataloader:

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src = torch.t(src)
        tgt = torch.t(tgt)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        # パラレルのために次元追加や転置をする
        src = torch.t(src)
        tgt_input = torch.t(tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        src_mask = src_mask.repeat(src.shape[0], 1, 1)
        tgt_mask = tgt_mask.repeat(tgt.shape[0], 1, 1)
        #print(src.size())
        #print(tgt_input.size())

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

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
        src = torch.t(src)
        tgt_input = torch.t(tgt_input)
        src_mask = src_mask.repeat(src.shape[0], 1, 1)
        tgt_mask = tgt_mask.repeat(tgt.shape[0], 1, 1)
        
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        if math.isnan(loss) == False:
            losses += loss.item()

    return losses / len(dataloader)


# 翻訳
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys


def load_data(batch_size):
    train_data_ja = torch.load("train_data/s_train_spm_ja.pt")
    test_data_ja = torch.load("train_data/s_test_spm_ja.pt")
    train_data_en = torch.load("train_data/s_train_spm_en.pt")
    test_data_en = torch.load("train_data/s_test_spm_en.pt")

    train_dataset = TensorDataset(train_data_ja, train_data_en)
    test_dataset = TensorDataset(test_data_ja, test_data_en)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader



def objective(trial):
    # ハイパーパラメータの設定
    BATCH_SIZE = int(trial.suggest_categorical('BATCH_SIZE', [4, 8, 32, 64]))
    NUM_EPOCHS = int(trial.suggest_categorical('NUM_EPOCHS', [4, 8, 32, 64]))
    NHEAD = int(trial.suggest_categorical('NHEAD', [4, 8, 16]))
    EMB_SIZE = int(trial.suggest_categorical('EMB_SIZE', [256, 512, 1024]))
    FFN_HID_DIM = int(trial.suggest_categorical('FFN_HID_DIM', [256, 512, 1024]))
    NUM_ENCODER_LAYERS = int(trial.suggest_categorical('NUM_ENCODER_LAYERS', [1, 3, 6]))
    NUM_DECODER_LAYERS = int(trial.suggest_categorical('NUM_DECODER_LAYERS', [1, 3, 6]))
    

    train_features = torch.load("train_features.pt")
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")
    valid_labels = torch.load("valid_labels.pt")

    SRC_VOCAB_SIZE = 16000 + 4
    TGT_VOCAB_SIZE = 16000 + 4
    PAD_IDX = 0
    torch.manual_seed(0)
    train_dataloader, test_dataloader = load_data(BATCH_SIZE)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer = torch.nn.DataParallel(transformer, device_ids=[0, 1, 2, 3])
    transformer = transformer.to(DEVICE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    val_loss_list = []
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        val_loss = evaluate(transformer, test_dataloader)
        val_loss_list.append(val_loss)

    return min(val_loss_list)


def output_optimal_param():
    # objectiveの出力がlossなので最小化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    # ベストパラメータを出力
    #print(study.best_params)
    #print(study.best_value)
    print(study.best_trial)
    return


output_optimal_param()

"""
実行結果
"""