import p30

#　指定された品詞の基本形を全てリストに入れて返す


def return_all_base(part_of_speech, file_input_name):
    all_part_of_speech_list = []
    sentence_mecab_list = p30.convert_mecab_to_list(file_input_name)
    for list in sentence_mecab_list:
        for dict in list:
            if dict.get('pos') == part_of_speech:
                all_part_of_speech_list.append(dict["base"])
    return all_part_of_speech_list


print(return_all_base("動詞", "neko.txt.mecab"))

'''
p.21 値の属性を追加する

実行結果 20個
['ある', '生れる', 'つく', 'する', '泣く', 'いる', 'する', 'いる', '始める', 'いう', '見る', '聞く', 'いう', 'ある', 'いう', '捕える', '煮る', '食う', 'いう', 'ある']
'''