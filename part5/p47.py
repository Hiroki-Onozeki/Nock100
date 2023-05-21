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
            is_trans = False
            is_wo = False
            verb = ""
            trans_phrase = ""

            for elem in elem_chunk["morphs"]:
                if elem["pos"] == "動詞":
                    verb = elem["base"]
                    is_verb = True
                    break
            
            # サ行変格接続名詞+をがかかるか判定する
            if is_verb == True:
                for index in elem_chunk["srcs"]:
                    for elem in sentence_list[int(index)]["morphs"]:
                        if elem["pos"] == "名詞" and  elem["pos1"] == "サ変接続":
                            is_trans = True
                        if elem["pos"] == "助詞" and  elem["surface"] == "を":
                            is_wo = True

                    if is_trans == True and is_wo == True:
                        for elem in sentence_list[int(index)]["morphs"]:
                            if elem["pos"] != "記号":
                                trans_phrase += elem["surface"]

            # 動詞にサ変接続が係っているなら、係っている文節の助詞をキーに、文節を値にして辞書に入れる
            if is_verb == True and is_trans == True and is_wo == True:
                for index in elem_chunk["srcs"]:
                    phrase = ""
                    for elem in sentence_list[int(index)]["morphs"]:
                        if elem["pos"] != "記号":
                            phrase += elem["surface"]
                        if elem["pos"] == "助詞":
                            particle = elem["base"]
                    particle_phrase_dict[particle] = phrase

                # 辞書のキーである助詞でソートしてから、リストに格納
                verb_particle_list.append(trans_phrase + verb)
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


make_verb_particle_file("trans_verb_particle_phrase_set.txt")


'''
p.21 値の属性を追加する

実行結果 
学習を行う	に を	元に 経験を

先頭２０行
cat trans_verb_particle_phrase_set.txt | head -n 20

知的行動を人間に代わる  に      を      人間に  知的行動を
推論判断をする  を      推論判断を
パターンをする  て      を      解析して        パターンを
記号処理を用いる        を      記号処理を
記述を主体とする        と      を      主体と  記述を
役割をする      に      を      計算機に        役割を
注目を集める    が      を      サポートベクターマシンが        注目を
経験を元に学習を行う    に      を      元に    学習を
流行を超える    を      流行を
学習を繰り返す  を      学習を
推論ルールを統計的学習を元に生成規則を通して生成するする        て      に      は      を      を通して        なされている        元に    ACT-Rでは       統計的学習を    生成する
進化を見せる    て      において        は      を      加えて  生成技術において        敵対的生成ネットワークは進化を
コンテンツ生成を行う    を      コンテンツ生成を
機械式計算機をする      は      を      1642年  機械式計算機を
開発を行った行う        は      を      エイダ・ラブレスは      行った
革命をもたらす  に      は      を      形式論理に      出版し  革命を
基礎を築いた築く        を      築いた
用語を作り出す  で      は      を      会議で  ジョンマッカーシーは    用語を
プログラミング言語をする        は      を      彼はまた        プログラミング言語を
テストをする    を      テストを
'''