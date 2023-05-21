from heapq import heappush, heappop
import torch
from torch import Tensor
import torch
import torch.nn as nn
import math
import numpy as np
import math
import sentencepiece as spm
from sacrebleu.metrics import BLEU
from matplotlib import pyplot


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


class BeamSearchNode(object):
    def __init__(self, wid, logp, length):
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



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


def beam_search_decoding(model, src, src_mask, max_len, sos_id, eos_id, beam_width):
    sequence_cand = []
    sequence_prob = []
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.module.encode(src, src_mask)
    memory = memory.to(DEVICE)
    word_id = torch.ones(1, 1).fill_(sos_id).type(torch.long).to(DEVICE)

    node = BeamSearchNode(wid=word_id, logp=0, length=1)
    nodes = []
    heappush(nodes, (node.eval(), id(node), node))

    best_node_list = []
    for i in range(max_len-1):
        # ビームサイズ分できればpopする
        if beam_width > len(nodes):
            pop_num = len(nodes)
        else:
            pop_num = beam_width

        for _ in range(pop_num):
            node = heappop(nodes)
            best_node_list.append(node)
            heapify(nodes)
        nodes = []

        for _, _, node in best_node_list:
            word_id = node.wid
            
            # デコードして確率を得る
            tgt_mask = (generate_square_subsequent_mask(word_id.size(0)).type(torch.bool)).to(DEVICE)
            out = model.module.decode(word_id, memory, tgt_mask)
            out = out.transpose(0, 1)
            out_prob = torch.softmax(model.module.generator(out[:, -1]), dim=1)

            # 確率が高いものからビームサイズ分取得
            prob_k, idx_k = torch.topk(out_prob, beam_width)
            prob_k = prob_k.squeeze(0)
            idx_k = idx_k.squeeze(0)

            # 取得したものをヒープに格納
            for j in range(len(prob_k)):
                prob = prob_k[j]
                next_word = idx_k[j]

                new_word = torch.cat([word_id, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                new_node = BeamSearchNode(wid=new_word, logp=node.logp+torch.log(prob).item(), length=node.length+1)

                if next_word == eos_id:
                    sequence_cand.append(new_word.flatten().cpu())
                    sequence_prob.append(new_node.eval())
                else:
                    heappush(nodes, (new_node.eval(), id(new_node), new_node))

    # 候補を取り出して、最も良いものを返す
    for i in range(beam_width - len(sequence_cand)):
        _, _, node = heappop(nodes)
        sequence_cand.append(node.wid.flatten().cpu())
        sequence_prob.append(node.eval())
    best_idx = np.argmax(sequence_prob)
    return sequence_cand[best_idx]


def translate_w_beam(model_pth, sentence, beam_width):
    # spmモデルを読み込む
    jasp = spm.SentencePieceProcessor()
    jasp.Load("spm_ja_sh.model")
    ensp = spm.SentencePieceProcessor()
    ensp.Load("spm_en_sh.model")

    # 文を整形する
    model = torch.load(model_pth)
    model.eval()
    id_list = jasp.EncodeAsIds(sentence)
    id_pt = torch.tensor(id_list)
    src = torch.unsqueeze(id_pt, dim=-1)
    num_tokens = src.shape[0]

    # デコードする
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_search_decoding(model,  src, src_mask, max_len=num_tokens + 5, sos_id=1, eos_id=2, beam_width=beam_width).flatten()
    ids = tgt_tokens.tolist()
    ids.pop(0)
    sentence = ensp.DecodeIds(ids)
    return sentence


# BLEUスコア算出
def calc_bleu_score(model_pth, beam_width):
    bleu = BLEU()
    refs = []
    hyps = [] 

    """
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
            hyps.append(translate_w_beam(model_pth, line, beam_width))
            n += 1
            if n > 5000:
                break

    result_train = bleu.corpus_score(hyps, refs)
    refs = []
    hyps = [] 
    """
    with open("./kftt-data-1.0/data/orig/kyoto-test.en")as f:
        ref = []
        for line in f:
            ref.append(line)
    refs.append(ref)
    
    # 仮設データ読み込み
    with open("./kftt-data-1.0/data/orig/kyoto-test.ja")as f:
        for line in f:
            hyps.append(translate_w_beam(model_pth, line, beam_width))

    result_test = bleu.corpus_score(hyps, refs)
    #print(result_train)
    print(str(result_test) + "\n")
    return 

def write_bleu_score():
    for i in range(1):
        beam_width = 64
        print("beam_width:" + str(beam_width) + "\n")
        calc_bleu_score('spm_model1_29ep.pth', beam_width)
    print("---done---")
    return

def plot_graph():
    beam_width = [1, 2, 4, 8, 16, 32]
    bleu_score = [10.57, 11.26, 11.88, 12.15, 12.12, 12.20]

    pyplot.plot(beam_width, bleu_score)
    pyplot.xlabel("beam_search_width")
    pyplot.ylabel("BLEU_score")
    pyplot.show()
    return

PAD_IDX = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(translate_w_beam('spm_model1_29ep.pth', "真筆であるか専門家の間でも意見の分かれるものも多々ある。", 2))


#calc_bleu_score('spm_model1_29ep.pth', 8)
#write_bleu_score()
plot_graph()

#nohup python3 -u p94.py > & output4.txt &

"""
実行結果

beam_width:1
BLEU = 10.57 33.7/13.8/6.9/3.9 (BP = 1.000 ratio = 1.027 hyp_len = 27287 ref_len = 26560)
beam_width:2
BLEU = 11.26 34.9/14.7/7.4/4.2 (BP = 1.000 ratio = 1.017 hyp_len = 27010 ref_len = 26560)
beam_width:4
BLEU = 11.88 35.8/15.5/7.9/4.5 (BP = 1.000 ratio = 1.007 hyp_len = 26756 ref_len = 26560)
beam_width:8
BLEU = 12.15 36.1/15.8/8.2/4.7 (BP = 1.000 ratio = 1.001 hyp_len = 26586 ref_len = 26560)
beam_width:16
BLEU = 12.12 36.2/15.9/8.1/4.6 (BP = 1.000 ratio = 1.006 hyp_len = 26710 ref_len = 26560)
beam_width:32
BLEU = 12.20 36.3/16.1/8.3/4.6 (BP = 0.999 ratio = 0.999 hyp_len = 26525 ref_len = 26560)
"""