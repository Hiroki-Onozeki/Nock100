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
            particle_list = []
            is_verb = False

            for elem in elem_chunk["morphs"]:
                if elem["pos"] == "動詞":
                    verb_particle_list.append(elem["base"])
                    is_verb = True
                    break
                
            # 動詞なら、係っている文節の助詞を抜き出してリストに入れる
            if is_verb == True:
                for index in elem_chunk["srcs"]:
                    for elem in sentence_list[int(index)]["morphs"]:
                        if elem["pos"] == "助詞":
                            particle_list.append(elem["base"])

                # 辞書順にソートしてから、動詞を含むリストに格納
                particle_list.sort()
                for particle in particle_list:
                    verb_particle_list.append(particle)

            if verb_particle_list != []:
                all_verb_particle_list.append(verb_particle_list)
        
    # リストの中身をタブ区切りにして一つの文字列に入れ、それをファイルに書き込む
    for list in all_verb_particle_list:
        contents += '\t'.join([word for word in list])
        contents += '\n'

    with open(output_file_name, "w") as w:
        w.write(contents)
    return

make_verb_particle_file("verb_particle_set.txt")

'''
P.67 要約コメント

実行結果
作り出す	で	は	を

出現頻度トップ１０
cat verb_particle_set.txt | sort | uniq -c | sort -r | head -n 10
sort -r 逆順にソート
uniq -c 出現回数を表示
head -n 先頭の行数指定

49 する       を
18 する       が
15 する       に
14 する       と
12 する       は      を
10 する       に      を
9 する       で      を
9 よる       に
8 行う       を
8 する

cat verb_particle_set.txt | grep "行う" | sort | uniq -c | sort -r 
grep 特定文字列のみ

   8 行う       を
   1 行う       まで    を
   1 行う       から
   1 行う       に      まで    を
   1 行う       は      を      をめぐって
   1 行う       に      に      により  を
   1 行う       て      に      は      は      は
   1 行う       が      て      で      に      は
   1 行う       が      で      に      は
   1 行う       に      を      を
   1 行う       で      に      を
   1 行う       て      に      を
   1 行う       て      て      を
   1 行う       が      で      は
   1 行う       は      を
   1 行う       で      を
   1 行う       て      に
   1 行う       に

cat verb_particle_set.txt | grep "なる" | sort | uniq -c | sort -r 

   3 なる       に      は
   3 なる       が      と
   2 なる       に
   2 なる       と
   1 無くなる   は
   1 異なる     が      で
   1 異なる     も
   1 なる       から    が      て      で      と      は
   1 なる       から    で      と
   1 なる       て      として  に      は
   1 なる       が      と      にとって        は
   1 なる       で      と      など    は
   1 なる       が      で      と      に      は      は
   1 なる       が      が      て      て      て      と
   1 なる       と      に      は      も
   1 なる       で      に      に      は
   1 なる       が      に      に      は
   1 なる       て      に      は
   1 なる       は      は
   1 なる       で      は
   1 なる       が      に
   1 なる       も
   1 なる

cat verb_particle_set.txt | grep "与える" | sort | uniq -c | sort -r

   1 与える     が      など    に
   1 与える     に      は      を
   1 与える     が      に
'''