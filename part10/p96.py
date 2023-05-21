from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sacrebleu.metrics import BLEU
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

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

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

    memory = model.module.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.module.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.module.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys


# 和英翻訳
def translate(model, sentence):
    jasp = spm.SentencePieceProcessor()
    jasp.Load("spm_ja_sh.model")
    ensp = spm.SentencePieceProcessor()
    ensp.Load("spm_en_sh.model")

    model.eval()
    id_list = jasp.EncodeAsIds(sentence)
    id_pt = torch.tensor(id_list)
    src = torch.unsqueeze(id_pt, dim=-1)

    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=0).flatten()
    
    sentence = ensp.DecodeIds(tgt_tokens.tolist())
    return sentence

# BLEUスコア算出
def calc_bleu_score(model):
    bleu = BLEU()
    refs = []
    hyps = [] 

    with open("./kftt-data-1.0/data/orig/kyoto-train.en")as f:
        ref = []
        n = 0
        for line in f:
            ref.append(line)
            n += 1
            if n > 5000:
                break
    refs.append(ref)
    
    # 仮設データ読み込み
    with open("./kftt-data-1.0/data/orig/kyoto-train.ja")as f:
        n = 0
        for line in f:
            hyps.append(translate(model, line))
            n += 1
            if n > 5000:
                break

    result_train = bleu.corpus_score(hyps, refs)
    refs = []
    hyps = [] 

    with open("./kftt-data-1.0/data/orig/kyoto-test.en")as f:
        ref = []
        for line in f:
            ref.append(line)
    refs.append(ref)
    
    # 仮設データ読み込み
    with open("./kftt-data-1.0/data/orig/kyoto-test.ja")as f:
        for line in f:
            hyps.append(translate(model, line))

    result_test = bleu.corpus_score(hyps, refs)
    result_train = float(str(result_train).split()[2])
    result_test = float(str(result_test).split()[2])
    return result_train, result_test


def train_model(output_file_name1, output_file_name2, train_dataloader, test_dataloader):
    print(str(NUM_EPOCHS) + "epochs")

    # tensorboard log dir 指定
    loss_writer = SummaryWriter(log_dir="./loss_logs")
    bleu_writer = SummaryWriter(log_dir="./bleu_logs")
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader)
        end_time = timer()
        val_loss = evaluate(transformer, test_dataloader)
        train_bleu, val_bleu = calc_bleu_score(transformer)

        # 書き込み
        loss_writer.add_scalars("loss", {"train_loss":train_loss, "val_loss":val_loss}, epoch)
        bleu_writer.add_scalars("BLEU", {"train_bleu":train_bleu, "val_loss":val_bleu}, epoch)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    
    loss_writer.close()
    bleu_writer.close()
    torch.save(transformer, output_file_name2)
    print("---done---")
    return

def load_data(batch_size):
    train_data_ja = torch.load("train_data/sh_train_spm_ja.pt")
    test_data_ja = torch.load("train_data/sh_test_spm_ja.pt")
    train_data_en = torch.load("train_data/sh_train_spm_en.pt")
    test_data_en = torch.load("train_data/sh_test_spm_en.pt")

    train_dataset = TensorDataset(train_data_ja, train_data_en)
    test_dataset = TensorDataset(test_data_ja, test_data_en)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader

BATCH_SIZE = 64
NUM_EPOCHS = 30
PAD_IDX = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader, test_dataloader = load_data(BATCH_SIZE)


torch.manual_seed(0)
SRC_VOCAB_SIZE = 16000 + 4
TGT_VOCAB_SIZE = 16000 + 4
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

transformer = torch.nn.DataParallel(transformer, device_ids=[0, 1, 2, 3])
transformer = transformer.to(DEVICE)


for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

train_model("model/p96_model_weight_30ep.pth", "model/p96_model_30ep.pth", train_dataloader, test_dataloader)

#nohup python3 -u p96.py > & output2.txt &
# tensorboard --logdir=loss_logs, bleu_logs

"""
実行結果

p96.png
"""

