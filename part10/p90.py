import MeCab
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# 単語とidのファイルを作成する
def make_data_file():
    ja_word_id_dict = {}
    ja_removed_word_id_dict = {}
    ja_sorted_word_id_dict = {}
    ja_train_id_list = []
    ja_test_id_list = []
    en_word_id_dict = {}
    en_removed_word_id_dict = {}
    en_sorted_word_id_dict = {}
    en_train_id_list = []
    en_test_id_list = []
    t_mecab = MeCab.Tagger('-Owakati')

    # 単語と出現頻度の辞書を作る
    with open("./kftt-data-1.0/data/orig/kyoto-train.ja") as f:
        for sentence in f:
            word_list = t_mecab.parse(sentence).split()
            for word in word_list:
                if word in ja_word_id_dict:
                    ja_word_id_dict[word] += 1
                else: 
                    ja_word_id_dict[word] = 1
    
    # 頻度が低いのものを削除
    for key, value in ja_word_id_dict.items():
        if value > 4:
            ja_removed_word_id_dict[key] = value
    
    # 辞書を値でソートして頻度順にidを付与する
    ja_sorted_word_id_list = sorted(ja_removed_word_id_dict.items(), key=lambda x:x[1], reverse=True)
    for i in range(len(ja_sorted_word_id_list)):
        ja_sorted_word_id_dict[ja_sorted_word_id_list[i][0]] = i+4
    
    with open("./kftt-data-1.0/data/orig/kyoto-train.ja") as f:
        for sentence in f:
            id_list = []
            words = t_mecab.parse(sentence).split()
            for word in words:
                if word in ja_sorted_word_id_dict.keys():
                    id_list.append(int(ja_sorted_word_id_dict[word]))
                else:
                    id_list.append(0)
            if len(id_list) > 32:
                id_list = id_list[:32]
            ja_train_id_list.append(id_list)
    
    with open("./kftt-data-1.0/data/orig/kyoto-tune.ja") as f:
        for sentence in f:
            id_list = []
            words = t_mecab.parse(sentence).split()
            for word in words:
                if word in ja_sorted_word_id_dict.keys():
                    id_list.append(int(ja_sorted_word_id_dict[word]))
                else:
                    id_list.append(0)
            if len(id_list) > 32:
                id_list = id_list[:32]
            ja_test_id_list.append(id_list)

    
    #　english
    with open("./kftt-data-1.0/data/orig/kyoto-train.en") as f:
        for sentence in f:
            word_list = t_mecab.parse(sentence).split()
            for word in word_list:
                if word in en_word_id_dict:
                    en_word_id_dict[word] += 1
                else: 
                    en_word_id_dict[word] = 1
    
    for key, value in en_word_id_dict.items():
        if value > 4:
            en_removed_word_id_dict[key] = value
    
    en_sorted_word_id_list = sorted(en_removed_word_id_dict.items(), key=lambda x:x[1], reverse=True)
    for i in range(len(en_sorted_word_id_list)):
        en_sorted_word_id_dict[en_sorted_word_id_list[i][0]] = i+4
    
    with open("./kftt-data-1.0/data/orig/kyoto-train.en") as f:
        for sentence in f:
            id_list = []
            words = t_mecab.parse(sentence).split()
            for word in words:
                if word in en_sorted_word_id_dict.keys():
                    id_list.append(int(en_sorted_word_id_dict[word]))
                else:
                    id_list.append(0)
            if len(id_list) > 32:
                id_list = id_list[:32]
            en_train_id_list.append(id_list)
    
    with open("./kftt-data-1.0/data/orig/kyoto-tune.en") as f:
        for sentence in f:
            id_list = []
            words = t_mecab.parse(sentence).split()
            for word in words:
                if word in en_sorted_word_id_dict.keys():
                    id_list.append(int(en_sorted_word_id_dict[word]))
                else:
                    id_list.append(0)
            if len(id_list) > 32:
                id_list = id_list[:32]
            en_test_id_list.append(id_list)

    ja_train_pt = [torch.tensor(x) for x in ja_train_id_list]
    ja_test_pt = [torch.tensor(x) for x in ja_test_id_list]
    en_train_pt = [torch.tensor(x) for x in en_train_id_list]
    en_test_pt = [torch.tensor(x) for x in en_test_id_list]
    ja_padded_train_pt = pad_sequence(ja_train_pt, batch_first=True, padding_value=1)
    ja_padded_test_pt = pad_sequence(ja_test_pt, batch_first=True, padding_value=1)
    en_padded_train_pt = pad_sequence(en_train_pt, batch_first=True, padding_value=1)
    en_padded_test_pt = pad_sequence(en_test_pt, batch_first=True, padding_value=1)
    print(ja_padded_train_pt.size())
    print(ja_padded_test_pt.size())
    print(en_padded_train_pt.size())
    print(en_padded_test_pt.size())
    print(len(ja_sorted_word_id_dict))
    print(len(en_sorted_word_id_dict))
    torch.save(ja_padded_train_pt, "train_data/s_dev_train_data_ja.pt")
    torch.save(ja_padded_test_pt, "train_data/s_dev_test_data_ja.pt")
    torch.save(en_padded_train_pt, "train_data/s_dev_train_data_en.pt")
    torch.save(en_padded_test_pt, "train_data/s_dev_test_data_en.pt")
    # 辞書を保存
    np.save("dev_ja_dict.npy", ja_sorted_word_id_dict)
    np.save("dev_en_dict.npy", en_sorted_word_id_dict)
    return 




make_data_file()
