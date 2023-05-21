import p41

# 係り元の文節と係り先の文節の文字をタブ区切りで出力する
def output_related_txt():
    all_sentence_list = []
    all_sentence_chunk = p41.input_txt_to_chunk()

    # １文ごとにリストに格納し、それを全体のリストに格納する
    for sentence_chunk in all_sentence_chunk:
        sentence_list = []

        # dstより、係り先の場所の文字列を取得して文にする
        for elem_chunk in sentence_chunk:
            sentence = ""
            index_num = elem_chunk["dst"]

            # 係り元の単語をまとめる
            for elem in elem_chunk["morphs"]:
                if elem["pos"] != "記号":
                    sentence += elem["surface"]
            sentence += "\t"

            # 係り先の単語をまとめる
            for elem in sentence_chunk[int(index_num)]["morphs"]:
                if elem["pos"] != "記号":
                    sentence += elem["surface"]
            sentence_list.append(sentence)
        all_sentence_list.append(sentence_list)
    return all_sentence_list


out_list = output_related_txt()[1]
for e in out_list:
    print(e)

'''
P.67 要約コメント

実行結果

人工知能        語
じんこうちのう  語
AI      エーアイとは
エーアイとは    語
計算    という
という  道具を
概念と  道具を
コンピュータ    という
という  道具を
道具を  用いて
用いて  研究する
知能を  研究する
研究する        計算機科学
計算機科学      の
の      一分野を
一分野を        指す
指す    語
語      研究分野とも
言語の  推論
理解や  推論
推論    問題解決などの
問題解決などの  知的行動を
知的行動を      代わって
人間に  代わって
代わって        行わせる
コンピューターに        行わせる
行わせる        技術または
技術または      研究分野とも
計算機  コンピュータによる
コンピュータによる      情報処理システムの
知的な  情報処理システムの
情報処理システムの      実現に関する
設計や  実現に関する
実現に関する    研究分野とも
研究分野とも    される
される  される
'''