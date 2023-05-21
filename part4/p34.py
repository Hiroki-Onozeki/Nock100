import p30

# 最も長く連続して出現する名詞を抽出する


def return_max_noun_sequence(file_input_name):
    max_sequence_length = 0
    max_sequence_str = ""

    # p30の関数を使用
    sentence_mecab_list = p30.convert_mecab_to_list(file_input_name)
    for list in sentence_mecab_list:
        is_sequence = False
        sequence_length = 0
        sequence_str = ""

        for dict in list:
            # フラッグを使用して、文内の最長連接名詞の長さと文字列を抽出し、これまででで最長だったら保存する
            if dict.get('pos') == "名詞" and is_sequence == True:
                sequence_length += 1
                sequence_str += dict["surface"]
            elif dict.get('pos') == "名詞" and is_sequence == False:
                is_sequence = True
                sequence_length = 1
                sequence_str = dict["surface"]
            elif dict.get('pos') != "名詞" and is_sequence == True:
                is_sequence = False
                if sequence_length > max_sequence_length:
                    max_sequence_length = sequence_length
                    max_sequence_str = sequence_str
                sequence_length = 0
                sequence_str = ""

        # 名詞の連接が途切れるか、文が終わったらこれまでの最長のと長さを比較
        if sequence_length > max_sequence_length:
            max_sequence_length = sequence_length
            max_sequence_str = sequence_str
    return max_sequence_str


print(return_max_noun_sequence("neko.txt.mecab"))

'''
P.67 要約コメント

実行結果
明治三十八年何月何日戸締り
'''
