import MeCab

# txtファイルをMecabで形態素解析し、結果を新たなファイルに書き込む


def write_file_w_mecab(file_input_name, file_output_name):
    tagger = MeCab.Tagger()
    with open(file_input_name) as f:
        contents_iunput_file = f.read()
    contents_mecab = tagger.parse(contents_iunput_file)

    with open(file_output_name, 'w') as w:
        w.write(contents_mecab)
    return


# 形態素解析の結果を整理して読み込み、リストに変換して返す
# なぜかデフォルト表示にならず、単語の全情報が表示されてしまう
def convert_mecab_to_list(file_input_name):
    all_sentence_mecab_list = []
    sentence_mecab_list = []

    with open(file_input_name) as f:
        for line in f:

            # 必要な情報を集める
            result_mecab_dict = {}
            # タブで分けると、表層系・それ以外の２要素になる
            result_mecab = line.split('\t')
            if len(result_mecab) > 1:
                result_mecab_dict["surface"] = result_mecab[0]
                tmp_result_mecab = result_mecab[1].split(',')
            # 必要な情報を辞書に格納し、文リストにそれを格納
            if len(tmp_result_mecab) > 6:
                result_mecab_dict["base"] = tmp_result_mecab[10]
                result_mecab_dict["pos"] = tmp_result_mecab[0]
                result_mecab_dict["pos1"] = tmp_result_mecab[1]
            sentence_mecab_list.append(result_mecab_dict)

            # 。か空白が来たらそれまでの文リストを全体のリストに格納
            if result_mecab[0] == "。" or result_mecab[0] == "\u3000":
                all_sentence_mecab_list.append(sentence_mecab_list)
                sentence_mecab_list = []

    return all_sentence_mecab_list


#write_file_w_mecab("neko.txt", "neko.txt.mecab")
'''
out_put_list = convert_mecab_to_list("neko.txt.mecab")
for e in out_put_list:
    print(e)
'''

'''
p.51 コードを段落に分割する

[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数詞'}, {'surface': '\u3000', 'base': '\u3000', 'pos': '空白', 'pos1': ''}]
[{'surface': '吾輩', 'base': '吾輩', 'pos': '代名詞', 'pos1': ''}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '猫', 'base': '猫', 'pos': '名詞', 'pos1': '普通名詞'}, {'surface': 'で', 'base': 'だ', 'pos': '助動詞', 'pos1': ''}, {'surface': 'ある', 'base': 'ある', 'pos': '動詞', 'pos1': '非自立可能'}, {'surface': '。', 'base': '。', 'pos': '補助記号', 'pos1': '句点'}]
[{'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '普通名詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': ''}, {'surface': '無い', 'base': '無い', 'pos': '形容詞', 'pos1': '非自立可能'}, {'surface': '。', 'base': '。', 'pos': '補助記号', 'pos1': '句点'}]
[{'surface': '\u3000', 'base': '\u3000', 'pos': '空白', 'pos1': ''}]
[{'surface': 'どこ', 'base': 'どこ', 'pos': '代名詞', 'pos1': ''}, {'surface': 'で', 'base': 'で', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': '生れ', 'base': '生れる', 'pos': '動詞', 'pos1': '一般'}, {'surface': 'た', 'base': 'た', 'pos': '助動詞', 'pos1': ''}, {'surface': 'か', 'base': 'か', 'pos': '助詞', 'pos1': '終助詞'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'んと', 'base': 'んと', 'pos': '感動詞', 'pos1': 'フィラー'}, {'surface': '見当', 'base': '見当', 'pos': '名詞', 'pos1': '普通名詞'}, {'surface': 'が', 'base': 'が', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'つか', 'base': 'つく', 'pos': '動詞', 'pos1': '非自立可能'}, {'surface': 'ぬ', 'base': 'ぬ', 'pos': '助動詞', 'pos1': ''}, {'surface': '。', 'base': '。', 'pos': '補助記号', 'pos1': '句点'}]
'''
