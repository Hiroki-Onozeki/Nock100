import p30
from matplotlib import pyplot as plt
import japanize_matplotlib

# 共起範囲は1文として、「猫」と共起頻度の高い単語トップ１０を棒グラフで表示する
def make_graph_co_occur(file_input_name):
    co_occur_dict = {}
    co_occur_str_list = []
    co_occur_freq_list = []
    freq_rank = []
    sentence_mecab_list = p30.convert_mecab_to_list(file_input_name)

    for list in sentence_mecab_list:
        is_co_occur = False
        # もしも文に「猫」が含まれていたらフラグを立てる
        for dict in list:
            if dict.get("base") == "猫":
                is_co_occur = True
        
        # フラグが立っているなら、文中の単語の共起回数を+1する
        if is_co_occur == True:
            for dict in list:
                word_base  = dict.get("base")
                if word_base in co_occur_dict.keys():
                    co_occur_dict[word_base] += 1
                else: 
                    co_occur_dict[word_base] = 1
    
    # 辞書の値を降順でソートして二次元リストに格納
    freq_co_occur_list = sorted(co_occur_dict.items(), key=lambda x: x[1], reverse=True)

    # 共起頻度トップ１０の単語の出現回数と単語を別々のリストに格納する
    for i in range(10):
        co_occur_str_list.append(freq_co_occur_list[i][0])
        co_occur_freq_list.append(freq_co_occur_list[i][1])
        freq_rank.append(i+1)
    
    # グラフに出力
    plt.bar(freq_rank, co_occur_freq_list, tick_label = co_occur_str_list)
    plt.title("「猫」共起頻度")
    plt.xlabel("単語")
    plt.ylabel("共起回数")
    plt.show()
    return

make_graph_co_occur("neko.txt.mecab")

'''
p.51 コードを段落に分割する

実行結果は　cat_co_occur_freq_graph.png
'''