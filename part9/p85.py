import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import part9.p80 as p80
from tqdm import tqdm
import gensim


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, label_num, weights, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(weights)
        #self.embedding = torch.nn.Embedding.from_pretrained(weights)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_dim*2, label_num)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x, x_len):
        # padding, packしてからlstmに送り、出力を解凍する
        x_len = x_len.cpu()
        emb_x = self.embedding(x)
        packed_emb_x = pack_padded_sequence(emb_x, x_len, enforce_sorted=False, batch_first=True)

        # 出力が双方向分２倍になる、ht,h1を使う   h0,c0はデフォルトで０
        packed_output, hc = self.lstm(packed_emb_x)
        h0t = torch.cat([hc[0][0], hc[0][1]], dim=1)
        y = self.linear(h0t)
        #y = self.softmax(y)
        return y



# モデルを学習させる
def train_model(b_size):
    train_features = torch.load("train_features.pt")#.int()
    train_labels = torch.load("train_labels.pt")
    valid_features = torch.load("valid_features.pt")#.int()
    valid_labels = torch.load("valid_labels.pt")
    train_len = torch.load("train_len.pt").int()
    valid_len = torch.load("valid_len.pt").int()
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
    train_dataset = TensorDataset(train_features, train_labels, train_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = TensorDataset(valid_features, valid_labels, valid_len)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        # モデルを訓練データで学習する
        sum_loss = 0
        acc_num = 0
        total = 0
        loss_num = 0
        model.train()

        bar = tqdm(total=len(train_dataloader))
        for x_train, y_train, x_len in train_dataloader:
            bar.update(1)
            # gpuに送る
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            pred = model(x_train, x_len)
            loss = loss_func(pred, y_train)
            loss.backward()
            opt.step()  #パラメータの更新
            opt.zero_grad()  #勾配を０にする

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
            for x_valid, y_valid, x_len in valid_dataloader:
                # gpuに送る
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                pred = model(x_valid, x_len)
                loss = loss_func(pred, y_valid)

                # 損失と正解個数を保存
                sum_loss += loss.item()
                pred_label = torch.argmax(pred, dim=1)
                acc_num += (pred_label == y_valid).sum().item()
                total += len(x_valid)
                loss_num += 1
        
         # エポックごとに平均損失と正解率をリストに格納
        valid_loss_list.append(sum_loss/loss_num)
        valid_acc_list.append(acc_num/total)
    
    print("-----result-----")
    print(train_loss_list)
    print(valid_loss_list)
    print(train_acc_list)
    print(valid_acc_list)
    return



# 重みを設定：out of memoryになるので置き換える
wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word_id_dict = p80.return_word_id_dict()
vocab_size = len(word_id_dict) + 1
weights = np.zeros((vocab_size, 300))
for idx, elem in enumerate(word_id_dict.keys()):  
    if elem in wv_model:
        weights[idx] = wv_model[elem]
    else:
        weights[idx] = np.random.randn(300)
weights = torch.from_numpy(np.array(weights).astype((np.float32)))

model = LSTM(vocab_size=vocab_size, hidden_dim=50, embedding_dim=300, label_num=4, weights=weights, num_layers=4)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_model(b_size=1)


"""
実行結果

batch_size=8, layer=1
[0.6787692471121115, 0.4219257457438728, 0.30304097521120216, 0.22686078984079522, 0.17754304595806494, 0.1417181073706995, 0.11191794369457013, 0.08895497388219338, 0.06980783688071268, 0.05317753649051892]
[0.5213921878063036, 0.42560981376888507, 0.35841328673020095, 0.3300452405687221, 0.3363823855315296, 0.3647815261181154, 0.38465769461064714, 0.4017376969993661, 0.4327723181494205, 0.4773497938559792]
[0.6409295352323838, 0.7504685157421289, 0.7872938530734632, 0.8172788605697151, 0.843328335832084, 0.8689092953523239, 0.8891491754122939, 0.9077023988005997, 0.9242878560719641, 0.9424662668665668]
[0.7241379310344828, 0.7518740629685158, 0.7856071964017991, 0.7983508245877061, 0.8020989505247377, 0.8148425787106447, 0.8238380809595203, 0.8305847076461769, 0.828335832083958, 0.8448275862068966]

batch_size=8, layer=2
[0.6970761398962144, 0.43009712582529486, 0.3115634431349005, 0.2322709698361279, 0.18352280941553067, 0.14501106198978328, 0.11281983484186803, 0.0940794693362158, 0.07243926673037385, 0.05459293218904339]
[0.5240163074995943, 0.43615362400601726, 0.3549344590502585, 0.34217230363297246, 0.3555169324418474, 0.35560384634910097, 0.3794502261886652, 0.36530293531654523, 0.37972726674979096, 0.4972320730068782]
[0.6290292353823088, 0.7464392803598201, 0.7859820089955023, 0.8161544227886057, 0.8422976011994003, 0.8670352323838081, 0.8887743628185907, 0.9055472263868066, 0.9233508245877061, 0.9394677661169415]
[0.717391304347826, 0.7578710644677661, 0.7773613193403298, 0.7901049475262368, 0.8043478260869565, 0.8148425787106447, 0.8200899550224887, 0.8260869565217391, 0.8305847076461769, 0.8335832083958021]

batch_size=8, layer=4
[0.6979561042448302, 0.433298426871558, 0.31284412152847385, 0.23756207977428564, 0.18770773956242173, 0.1508533436301649, 0.11789131680751067, 0.09407864168597957, 0.07503498470848854, 0.05866862010278756]
[0.542819566177037, 0.4431775893101435, 0.38229326261374763, 0.3553714868640471, 0.35749804259495377, 0.37471426165701743, 0.3818463419848484, 0.4196654117392923, 0.4367671177240122, 0.5317525673069218]
[0.6289355322338831, 0.7446589205397302, 0.7853260869565217, 0.8089392803598201, 0.837612443778111, 0.8618815592203898, 0.8845577211394303, 0.901424287856072, 0.9160419790104948, 0.9353448275862069]
[0.7031484257871065, 0.7481259370314842, 0.7736131934032984, 0.7811094452773614, 0.795352323838081, 0.8088455772113943, 0.8148425787106447, 0.8193403298350824, 0.8305847076461769, 0.8238380809595203]

batch_size=1, layer=1
[0.7285672333200773, 0.30649875791528447, 0.10161863456022267, 0.030335651167229506, 0.010818923390043929, 0.005257356643324694, 0.005438399359068142, 0.0048604424255245466, 0.003564649038981699, 0.0031975754326504114]
[0.512079860117586, 0.4115835578823836, 0.48080211466590433, 0.5218522951227059, 0.5697094182967725, 0.5758273852398494, 0.5912417937899979, 0.5930959488195988, 0.6126736201417974, 0.6231707905864798]
[0.736319340329835, 0.8954272863568216, 0.9642053973013494, 0.9906296851574213, 0.9974700149925038, 0.9987818590704648, 0.998688155922039, 0.9985944527736131, 0.9988755622188905, 0.9989692653673163]
[0.8140929535232384, 0.8643178410794603, 0.8620689655172413, 0.881559220389805, 0.876311844077961, 0.8868065967016492, 0.8883058470764618, 0.8845577211394303, 0.883808095952024, 0.8845577211394303]

batch_size=1, layer=4
[0.7320815448699725, 0.30514087153047653, 0.10548503009468896, 0.030085324356914554, 0.01311543991947732, 0.005786333514507302, 0.005467130014768321, 0.004261153785492046, 0.003958798568617623, 0.0034542925294463803]
[0.538073635822921, 0.43448768621964695, 0.48415988343203575, 0.6092694371133434, 0.6409547635972429, 0.6743612297582913, 0.6836932719068742, 0.6913865666689789, 0.6961764056530699, 0.7181610750162982]
[0.7336019490254873, 0.8944902548725637, 0.9628935532233883, 0.9925974512743628, 0.9970952023988006, 0.9984070464767616, 0.9983133433283359, 0.9984070464767616, 0.9985007496251874, 0.9985007496251874]
[0.8035982008995503, 0.8575712143928036, 0.8665667166416792, 0.8680659670164917, 0.8748125937031485, 0.8778110944527736, 0.8778110944527736, 0.8785607196401799, 0.876311844077961, 0.876311844077961]
"""