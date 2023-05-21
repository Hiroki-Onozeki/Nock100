import p35
from matplotlib import pyplot as plt
import japanize_matplotlib

# 単語の出現頻度のヒストグラムを表示する


def make_histgram_word_freq(file_input_name):
    freq_dict = {}
    freq_num_list = []
    freq_freq_list = []
    freq_rank = []
    word_freq_list = p35.return_sort_word_frequency_list(file_input_name)

    # 単語出現頻度の値の個数をカウントして辞書に入れる
    for list in word_freq_list:
        if list[1] in freq_dict.keys():
            freq_dict[list[1]] += 1
        else:
            freq_dict[list[1]] = 1

    # ソートする必要はないが、確認時に便利なため行う
    freq_dict = sorted(freq_dict.items(), reverse=True)

    # 共起頻度トップ１０の単語の出現回数と単語を別々のリストに格納する
    for elem in freq_dict:
        freq_num_list.append(elem[0])
        freq_freq_list.append(elem[1])

    # グラフに出力
    plt.bar(freq_num_list, freq_freq_list)
    plt.title("単語出現頻度のヒストグラム")
    plt.xlabel("単語出現頻度の値")
    plt.ylabel("単語の種類数")
    plt.xlim(0, 9547)
    plt.show()
    return


make_histgram_word_freq("neko.txt.mecab")


'''
p.51 コードを段落に分割する

実行結果　word_freq_histgram.png
見やすくするためにx,yの範囲に制限したもの　word_freq_histgram_limited.png
'''
