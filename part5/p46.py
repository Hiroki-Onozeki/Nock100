import p41

# 動詞に係っている助詞をセットでファイルに出力する
def make_verb_particle_file(output_file_name):
    all_sentence_list = p41.input_txt_to_chunk()
    all_verb_particle_list = []
    contents = ""
    for sentence_list in all_sentence_list:

        # 動詞を含む文節かを判定し、最も左の動詞を格納する
        for elem_chunk in sentence_list:
            verb_particle_list = []
            particle_phrase_dict = {}
            is_verb = False

            for elem in elem_chunk["morphs"]:
                if elem["pos"] == "動詞":
                    verb_particle_list.append(elem["base"])
                    is_verb = True
                    break
                
            # 動詞なら、係っている文節の助詞をキーに、文節を値にして辞書に入れる
            if is_verb == True:
                for index in elem_chunk["srcs"]:
                    phrase = ""
                    for elem in sentence_list[int(index)]["morphs"]:
                        if elem["pos"] != "記号":
                            phrase += elem["surface"]
                        if elem["pos"] == "助詞":
                            particle = elem["base"]
                    particle_phrase_dict[particle] = phrase

                # 辞書のキーである助詞でソートしてから、動詞を含むリストに格納
                sorted_particle_phrase_list = sorted(particle_phrase_dict.items())
                for set_list in sorted_particle_phrase_list:
                    verb_particle_list.append(set_list[0])
                for set_list in sorted_particle_phrase_list:
                    verb_particle_list.append(set_list[1])

            if verb_particle_list != []:
                all_verb_particle_list.append(verb_particle_list)
        
    # リストの中身をタブ区切りにして一つの文字列に入れ、それをファイルに書き込む
    for list in all_verb_particle_list:
        contents += '\t'.join([word for word in list])
        contents += '\n'

    with open(output_file_name, "w") as w:
        w.write(contents)
    return

make_verb_particle_file("verb_particle_phrase_set.txt")

'''
p.16 抽象的な名前よりも具体的な名前を使う

実行結果 
作り出す	で	は	を	会議で	ジョンマッカーシーは	作り出した

先頭２０行
cat verb_particle_phrase_set.txt | head -n 20

用いる  を      道具を
する    て      を      用いて  知能を
指す    を      一分野を
代わる  に      を      人間に  知的行動を
行う    て      に      代わって        コンピューターに
する    も      研究分野とも
述べる  で      に      は      解説で  次のように      佐藤理史は
する    で      を      コンピュータ上で        知的能力を
する    を      推論判断を
する    を      画像データを
する    て      を      解析して        パターンを
'''