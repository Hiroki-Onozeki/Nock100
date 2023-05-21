import p41

# 名詞を含む文節が動詞を含む文節に係るとき、タブ区切りで出力する
def output_related_txt():
    all_sentence_list = []
    all_sentence_chunk = p41.input_txt_to_chunk()

    # １文ごとにリストに格納し、それを全体のリストに格納する
    for sentence_chunk in all_sentence_chunk:
        sentence_list = []

        # dstより、係り先の場所の文字列を取得し、係りもとに名詞を含むか・係り先に動詞を含むかをフラグで判別し、リストに文を追加する
        for elem_chunk in sentence_chunk:
            sentence = ""
            is_noun = False
            is_verb = False
            index_num = elem_chunk["dst"]

            #　係り元の単語をまとめる
            for elem in elem_chunk["morphs"]:
                if elem["pos"] != "記号":
                    sentence += elem["surface"]
                    if elem["pos"] == "名詞":
                        is_noun = True
            sentence += "\t"

            # 係り先の単語をまとめる
            for elem in sentence_chunk[int(index_num)]["morphs"]:
                if elem["pos"] != "記号":
                    sentence += elem["surface"]
                    if elem["pos"] == "動詞":
                        is_verb = True
            
            if is_verb == True and is_noun == True:
                sentence_list.append(sentence)
        all_sentence_list.append(sentence_list)
    return all_sentence_list


out_list = output_related_txt()[1]
for e in out_list:
    print(e)

'''
p.21 値の属性を追加する

実行結果

道具を  用いて
知能を  研究する
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される
'''