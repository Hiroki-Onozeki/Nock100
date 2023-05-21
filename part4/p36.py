import p35
from matplotlib import pyplot as plt
import japanize_matplotlib

# 出現頻度トップ１０の出現頻度を棒グラフで表示する
def make_graph_word_frequency(title_input_name):
    word_base_str_list = []
    word_freq_list = []
    left = []

    word_freq_str_list = p35.return_sort_word_frequency_list(title_input_name)

    # 出現頻度トップ１０の単語の出現回数と単語を別々のリストに格納する
    for i in range(10):
        word_base_str_list.append(word_freq_str_list[i][0])
        word_freq_list.append(word_freq_str_list[i][1])
        left.append(i+1)
    
    # グラフに出力
    plt.bar(left, word_freq_list, tick_label = word_base_str_list)
    plt.title("単語出現頻度")
    plt.xlabel("単語")
    plt.ylabel("出現回数")
    plt.show()
    return

make_graph_word_frequency("neko.txt.mecab")

'''
P.67 要約コメント

実行結果は word_freq_graph.png
'''
