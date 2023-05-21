import p41
import time

# 名詞句のペアを結ぶ最短係り受けパスを抽出する
def make_shortest_noun_pharase_pass_file(output_file_name):
    all_sentence_list = p41.input_txt_to_chunk("john_parsed.txt")
    
    for sentence_list in all_sentence_list:
        all_noun_phrase_list = []
        index = 0

        for elem_chunk in sentence_list:
            is_noun = False
            noun_phrase = ""
            noun_phrase_list = []

            # 文節に名詞を含んでいるか判別する
            for elem in elem_chunk["morphs"]:
                if elem["pos"] == "名詞":
                    dst = elem_chunk["dst"]
                    srcs = elem_chunk["srcs"]
                    is_noun = True
                        
            # 名詞を含んでいるなら、名詞をxに変換し、文節・dst・srcsをまとめてリストに格納する
            if is_noun == True:
                is_x = False
                for elem in elem_chunk["morphs"]:
                    if elem["pos"] != "記号" and elem["pos"] != "名詞":
                        noun_phrase += elem["surface"]
                    elif elem["pos"] == "名詞":
                        if is_x == False:
                            noun_phrase += 'x'
                            is_x = True
                noun_phrase_list.append(noun_phrase)
                noun_phrase_list.append(index)
                noun_phrase_list.append(dst)
                noun_phrase_list.append(srcs)
                all_noun_phrase_list.append(noun_phrase_list)
        
            index += 1

        # 名詞リストから2つ選んで名詞部分をXYに変換する
        for i in range(len(all_noun_phrase_list)):
            noun_x_list = all_noun_phrase_list[i]
            noun_x_phrase = noun_x_list[0]
            noun_x_phrase = noun_x_phrase.replace('x', "X")
            for j in range(len(all_noun_phrase_list)-i-1):
                is_on_pass = False
                is_cross_pass = False
                noun_y_list = all_noun_phrase_list[i+j+1]
                noun_y_phrase = noun_y_list[0]
                noun_y_phrase = noun_y_phrase.replace('x', "Y")
                
                # 経路上かどうか判断
                dst_index = noun_x_list[2]
                while int(dst_index) > 0:
                    if int(dst_index) == int(noun_y_list[1]):
                        is_on_pass = True
                        break
                    else: 
                        dst_index = sentence_list[int(dst_index)]["dst"]
                        
                # 経路上ならXからYまでのパスを表示
                if is_on_pass == True:
                    print(noun_x_phrase, end='')
                    dst_index = noun_x_list[2]
                    
                    while int(dst_index) != int(noun_y_list[1]):
                        phrase = ""
                        for elem in sentence_list[int(dst_index)]["morphs"]:
                            if elem["pos"] != "記号":
                                phrase += elem["surface"]
                        print(" -> " + phrase, end='')
                        dst_index = sentence_list[int(dst_index)]["dst"]
                    print(" -> " + noun_y_phrase)

                # 経路上でないなら、yから根へのパスのうち、xと合流する点があるか判断し、その地点を保存する
                else:
                    y_dst_index = int(noun_y_list[2])
                    while y_dst_index > 0:
                        x_dst_index = int(noun_x_list[2])
                        while int(x_dst_index) > 0:
                            if x_dst_index == y_dst_index:
                                is_cross_pass = True
                                target_index = x_dst_index
                                break
                            x_dst_index = int(sentence_list[x_dst_index]["dst"])
                        y_dst_index = int(sentence_list[y_dst_index]["dst"])
                    
                    # xとyの合流点があるなら、xから合流地点までのパスを表示
                    if is_cross_pass == True:
                        x_dst_index = int(noun_x_list[2])
                        print(noun_x_phrase, end='')

                        while x_dst_index != target_index:
                            phrase = ""
                            for elem in sentence_list[x_dst_index]["morphs"]:
                                if elem["pos"] != "記号":
                                    phrase += elem["surface"]
                            print(" -> " + phrase, end='')
                            x_dst_index = int(sentence_list[x_dst_index]["dst"])
                        
                        # yから合流地点までのパスを表示
                        y_dst_index = int(noun_y_list[2])
                        print(" | " + noun_y_phrase, end='')
                        while y_dst_index != target_index:
                            phrase = ""
                            for elem in sentence_list[y_dst_index]["morphs"]:
                                if elem["pos"] != "記号":
                                    phrase += elem["surface"]
                            print(" -> " + phrase, end='')
                            y_dst_index = int(sentence_list[y_dst_index]["dst"])

                        # 合流地点を表示
                        phrase = ""
                        for elem in sentence_list[target_index]["morphs"]:
                                if elem["pos"] != "記号":
                                    phrase += elem["surface"]
                        print(" | " + phrase)

    return


make_shortest_noun_pharase_pass_file("noun_noun_pass.txt")

'''
P.67 要約コメント

実行結果

Xは | Yに関する -> 最初の -> 会議で | 作り出した
Xは | Yの -> 会議で | 作り出した
Xは | Yで | 作り出した
Xは | Yという -> 用語を | 作り出した
Xは | Yを | 作り出した
Xに関する -> Yの
Xに関する -> 最初の -> Yで
Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
Xに関する -> 最初の -> 会議で | Yを | 作り出した
Xの -> Yで
Xの -> 会議で | Yという -> 用語を | 作り出した
Xの -> 会議で | Yを | 作り出した
Xで | Yという -> 用語を | 作り出した
Xで | Yを | 作り出した
Xという -> Yを
'''