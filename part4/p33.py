import p30

#２つの名詞がので連結されている名詞句を抽出したリストを返す


def return_noun_phrase(file_input_name):
    noun_phrase_list = []
    # 問題３０の関数を使う
    sentence_mecab_list = p30.convert_mecab_to_list(file_input_name)

    #名詞+の+名詞になっていればリストに追加
    for list in sentence_mecab_list:
        for i in range(len(list)-2):
            if list[i].get('pos') == "名詞" and list[i+1].get('surface') == "の" and list[i+2].get('pos') == '名詞':
                noun_phrase_list.append(list[i]['surface']+list[i+1]['surface']+list[i+2]['surface'])
    return noun_phrase_list[:20]


print(return_noun_phrase("neko.txt.mecab"))

'''
p.21 値の属性を追加する

実行結果
['掌の上', '書生の顔', 'はずの顔', '顔の真中', '穴の中', '書生の掌', '掌の裏', '藁の上', '笹原の中', '池の前', '一樹の蔭', '垣根の穴', '隣家の三毛', '時の通路', '一刻の猶予', '家の内', '以外の人間', '前の書生', '胸の痞', '家の主人']
'''