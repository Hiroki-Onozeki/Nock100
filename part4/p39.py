import p35
from matplotlib import pyplot as plt
import japanize_matplotlib

# 単語の出現頻度順位とその出現回数の両対数グラフを表示する


def make_word_freq_rank_graph(file_input_name):
    freq_num_list = []
    freq_rank_list = []
    rank = 0
    word_freq_list = p35.return_sort_word_frequency_list(file_input_name)

    # 順位と出現回数を別々のリストに格納
    for list in word_freq_list:
        rank += 1
        freq_rank_list.append(rank)
        freq_num_list.append(list[1])

    # グラフに出力 scatterでデータをプロットし、plotで折れ線グラフにする
    plt.scatter(freq_rank_list, freq_num_list)
    plt.plot(freq_rank_list, freq_num_list)
    # x,yともに10の対数の軸に設定
    plt.loglog(basex=10, basey=10)
    plt.title("単語出現頻度順位と出現回数の関係")
    plt.xlabel("単語出現頻度順位")
    plt.ylabel("出現回数")
    plt.show()
    return


make_word_freq_rank_graph("neko.txt.mecab")


'''
p.63 質問されそうなことを想像する

実行結果　word_freq_rank_graph_png
'''
