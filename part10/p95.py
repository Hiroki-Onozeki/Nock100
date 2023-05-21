import sentencepiece as spm
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from timeit import default_timer as timer
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sacrebleu.metrics import BLEU


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


# spmモデルを学習し保存
def train_spm_model():
    spm.SentencePieceTrainer.train(
        '--input=kftt-data-1.0/data/orig/kyoto-train.ja, --model_prefix=spm_en_sh --character_coverage=0.9995 --vocab_size=8000 --pad_id=3'
        )
    return

# 学習用ファイル作成
def make_spm_file():
    sp = spm.SentencePieceProcessor()
    sp.Load("spm_en_sh.model")
    train_id_list = []
    test_id_list = []
    #print(sp.GetPieceSize())

    with open("./kftt-data-1.0/data/orig/kyoto-train.en")as f:
        for line in f:
            id_list = sp.EncodeAsIds(line)
            if len(id_list) >= 33:
                id_list = id_list[:32]
            train_id_list.append(id_list)

    with open("./kftt-data-1.0/data/orig/kyoto-test.en")as f:
        for line in f:
            id_list = sp.EncodeAsIds(line)
            if len(id_list) >= 33:
                id_list = id_list[:32]
            test_id_list.append(id_list)

    train_pt = [torch.tensor(x) for x in train_id_list]
    test_pt = [torch.tensor(x) for x in test_id_list]
    padded_train_pt = pad_sequence(train_pt, batch_first=True, padding_value=0)
    padded_test_pt = pad_sequence(test_pt, batch_first=True, padding_value=0)

    print(padded_train_pt.size())
    print(padded_test_pt.size())
    torch.save(padded_train_pt, "train_data/sh_train_spm_en.pt")
    torch.save(padded_test_pt, "train_data/sh_test_spm_en.pt")
    return

# padid確認
def output_pad_id():
    sp = spm.SentencePieceProcessor()
    sp.Load("spm_ja.model")
    print(sp.PieceToId('<pad>'))
    print(sp.PieceToId('<unk>'))
    #print(sp.IdToPiece())
    return



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
def translate(model_pth, sentence):
    jasp = spm.SentencePieceProcessor()
    jasp.Load("spm_ja_sh.model")
    ensp = spm.SentencePieceProcessor()
    ensp.Load("spm_en_sh.model")

    model = torch.load(model_pth)
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
def calc_bleu_score(model_pth):
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
            hyps.append(translate(model_pth, line))
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
            hyps.append(translate(model_pth, line))

    result_test = bleu.corpus_score(hyps, refs)
    #result_train = float(str(result_train).split()[2])
    #result_test = float(str(result_test).split()[2])
    print(result_train)
    print(result_test)
    return 

#train_spm_model()
#make_spm_file()
#output_pad_id()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(translate('model/spm_model_30ep.pth', "朱鞘の木刀を差すなど、風変わりな格好をして街を歩きまわった。"))
#print(translate('sp_model/spm_model1_1ep.pth', "真筆であるか専門家の間でも意見の分かれるものも多々ある。"))
calc_bleu_score('model/spm_model_30ep.pth')


"""
実行結果
臨済宗は、その名の通り、会昌の廃仏後、唐末の宗祖臨済義玄に始まる。
Rinzai Rinzai sect was founded after the Buddhist sect, the Rinzai sect, the founder of the So-ho-ji Temple, the founder of the So

朱鞘の木刀を差すなど、風変わりな格好をして街を歩きまわった。
like saya wooden swords with shusya wooden swordss, etc, and walked along with a wind---formed-f

BLEUスコア
train: BLEU = 10.44 36.3/14.7/7.3/4.1 (BP = 0.928 ratio = 0.930 hyp_len = 120408 ref_len = 129406)
test:  BLEU = 9.58 31.9/12.7/6.1/3.4 (BP = 1.000 ratio = 1.027 hyp_len = 27289 ref_len = 26560)
"""