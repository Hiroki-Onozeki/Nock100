import p41

# 名詞から根へのパスを抽出する
def make_verb_particle_file(output_file_name):
    all_sentence_list = p41.input_txt_to_chunk()
    all_noun_phrase_pass_list = []
    contents = ""
    for sentence_list in all_sentence_list:

        # パスを全体のリストに格納する
        for elem_chunk in sentence_list:
            noun_phrase_pass_list = []
            is_noun = False
            noun_phrase = ""

            # 文節に名詞を含んでいるか判別する
            for elem in elem_chunk["morphs"]:
                if elem["pos"] == "名詞":
                    verb = elem["base"]
                    is_noun = True
                    dst_index = elem_chunk["dst"]
                    break
                        
            # 名詞を含んでいるなら、その文節をリストに格納し、構文木の根まで文節を格納していく
            if is_noun == True:
                for elem in elem_chunk["morphs"]:
                    if elem["pos"] != "記号":
                            noun_phrase += elem["surface"]
                noun_phrase_pass_list.append(noun_phrase)

                # 根まで行くとdstは−１になる
                while int(dst_index) > 0:
                    noun_phrase = ""
                    for elem in sentence_list[int(dst_index)]["morphs"]:
                        if elem["pos"] != "記号":
                            noun_phrase += elem["surface"]
                    noun_phrase_pass_list.append(noun_phrase)
                    dst_index = sentence_list[int(dst_index)]["dst"]

                all_noun_phrase_pass_list.append(noun_phrase_pass_list)


        
    # リストの中身をタブ区切りにして一つの文字列に入れ、それをファイルに書き込む
    for list in all_noun_phrase_pass_list:
        contents += ' -> '.join([word for word in list])
        contents += '\n'

    with open(output_file_name, "w") as w:
        w.write(contents)
    return


make_verb_particle_file("noun_phrase_pass.txt")


'''
P.67 要約コメント

実行結果
ジョンマッカーシーは -> 作り出した
AIに関する -> 最初の -> 会議で -> 作り出した
最初の -> 会議で -> 作り出した
会議で -> 作り出した
人工知能という -> 用語を -> 作り出した
用語を -> 作り出した

先頭２０行
cat noun_phrase_pass.txt | head -n 20

人工知能
人工知能 -> 語 -> 研究分野とも -> される
じんこうちのう -> 語 -> 研究分野とも -> される
AI -> エーアイとは -> 語 -> 研究分野とも -> される
エーアイとは -> 語 -> 研究分野とも -> される
計算 -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
概念と -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
コンピュータ -> という -> 道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
道具を -> 用いて -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
知能を -> 研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
研究する -> 計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
計算機科学 -> の -> 一分野を -> 指す -> 語 -> 研究分野とも -> される
一分野を -> 指す -> 語 -> 研究分野とも -> される
語 -> 研究分野とも -> される
言語の -> 推論 -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
理解や -> 推論 -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
推論 -> 問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
問題解決などの -> 知的行動を -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
知的行動を -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
人間に -> 代わって -> 行わせる -> 技術または -> 研究分野とも -> される
'''