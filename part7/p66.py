import p61
from tqdm import tqdm


# コサイン類似度を求め、追加したファイルを作成する
def make_add_cosine_sim():

    bar = tqdm(total = 353)
    with open("wordsim353/combined.tab") as f:
        for line in f:
            words_score_list = line.split()
            # 単語１、単語２、人間スコア、コサイン類似度の順にリストに入れる
            if len(words_score_list) == 3:
                words_cosine_sim = p61.output_cosine_similarity(words_score_list[0], words_score_list[1])
                words_score_list.append(str(words_cosine_sim))

                # 一度ファイルに書き出す
                with open("conbined_w_cosine.tab", "a") as w:
                    w.write("\t".join(words_score_list))
                    w.write('\n')
                bar.update(1)

    return


# スピアマン相関係数を求める
def calculate_spearman_correlation():
    all_words_score_list = []
    with open("conbined_w_cosine.tab") as f:
        for line in f:
            words_score_list = line.split()
            all_words_score_list.append(words_score_list)
    
    # 人間スコア、コサイン類似度でそれぞれソートして新たなリストを作成する int!!!!!!!
    sorted_human_list = sorted(all_words_score_list, key=lambda x: x[2], reverse=True)
    sorted_cosine_list = sorted(all_words_score_list, key=lambda x: x[3], reverse=True)

    # ソート結果を元に、順位を元のリストに追加していく
    rank = 1
    for elem in sorted_human_list:
        for ws_list in all_words_score_list:
            if elem[0] == ws_list[0] and elem[1] == ws_list[1]:
                ws_list.append(rank)
                rank += 1
                continue
    
    rank = 1
    for elem in sorted_cosine_list:
        for ws_list in all_words_score_list:
            if elem[0] == ws_list[0] and elem[1] == ws_list[1]:
                ws_list.append(rank)
                rank += 1
                continue

    # スピアマン相関係数を求める
    sum_d2 = 0
    n = 0
    for elem in all_words_score_list:
        d = elem[4] - elem[5]
        sum_d2 += d**2
        n += 1
    spear_coef = 1 - ((6*sum_d2) / (n*(n**2-1)))

    return spear_coef

print(calculate_spearman_correlation())
#make_add_cosine_sim()

'''
実行結果

0.6841447072637354
'''