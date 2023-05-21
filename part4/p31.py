import p30

#　指定された品詞の表層形を全てリストに入れて返す


def return_all_surface(part_of_speech, file_input_name):
    all_part_of_speech_list = []
    sentence_mecab_list = p30.convert_mecab_to_list(file_input_name)
    for list in sentence_mecab_list:
        for dict in list:
            if dict.get('pos') == part_of_speech:
                all_part_of_speech_list.append(dict["surface"])
    return all_part_of_speech_list


print(return_all_surface("動詞", "neko.txt.mecab"))

'''
p.21 値の属性を追加する

実行結果 20個
['ある', '生れ', 'つか', 'し', '泣い', 'い', 'し', 'いる', '始め', 'いう', '見', '聞く', 'いう', 'あっ', 'いう', '捕え', '煮', '食う', 'いう', 'ある']
'''
