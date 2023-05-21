import p40

# 文節を表すクラス
class Chunk:
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []

# テキストの係り受け解析結果を読み込んで、文節の文字列と係り先を出力する
def input_txt_to_chunk(input_file_name="ai.ja.txt.parsed"):
    morphs_list = []
    sentence_chunk = []
    all_sentence_chunk = []
    with open(input_file_name) as f:
        # morphs dstに値を入れて、1文ずつのリストを全体のリストに格納する
        for line in f:
            cabocha_sentence = line.split()
            
            # dstを抜き出し、Chunkクラスを一度しまう
            if cabocha_sentence[0] == "*":
                if morphs_list != []:
                    result_chunk = Chunk(morphs_list, dst, "")
                    sentence_chunk.append(vars(result_chunk))
                    morphs_list = []
                dst = cabocha_sentence[2].replace("D", "")
            
            # morphオブジェクトのリストを抜き出す
            elif cabocha_sentence[0] != "*" and cabocha_sentence[0] != "EOS":
                cabocha_word = cabocha_sentence[1].split(",")
                result_morph = p40.Morph(cabocha_sentence[0], cabocha_word[6], cabocha_word[0], cabocha_word[1])
                morphs_list.append(vars(result_morph))

            #　Chunkをしまい、１分のリストを全体のリストに格納
            elif cabocha_sentence[0] == "EOS":
                if morphs_list != []:
                    result_chunk = Chunk(morphs_list, dst, "")
                    sentence_chunk.append(vars(result_chunk))
                    morphs_list = []
                    # 別関数でsrcsを入力する
                    all_sentence_chunk.append(add_index(sentence_chunk))
                    sentence_chunk = []

    return all_sentence_chunk

# Chunkクラスのdstから、srcを入力する
def add_index(sentence_list):
    for i in range(len(sentence_list)):
        index_num = sentence_list[i]["dst"]
        if int(index_num) >= 0:
            sentence_list[int(index_num)]["srcs"].append(i)
    return sentence_list

'''
res_list = input_txt_to_chunk("ai.ja.txt.parsed")[1]
for elem in res_list:
    print(elem)
'''


'''
P.67 要約コメント

実行結果
{'morphs': [{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}, {'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}], 'dst': '17', 'srcs': []}
{'morphs': [{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}, {'surface': 'じん', 'base': 'じん', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'こうち', 'base': 'こうち', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'のう', 'base': 'のう', 'pos': '助詞', 'pos1': '終助詞'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}], 'dst': '17', 'srcs': []}
{'morphs': [{'surface': 'AI', 'base': '*', 'pos': '名詞', 'pos1': '一般'}], 'dst': '3', 'srcs': []}
{'morphs': [{'surface': '〈', 'base': '〈', 'pos': '記号', 'pos1': '括弧開'}, {'surface': 'エーアイ', 'base': '*', 'pos': '名詞', 'pos1': '固有名詞'}, {'surface': '〉', 'base': '〉', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}], 'dst': '17', 'srcs': [2]}
{'morphs': [{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}], 'dst': '5', 'srcs': []}
{'morphs': [{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '9', 'srcs': [4]}
{'morphs': [{'surface': '概念', 'base': '概念', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '並立助詞'}], 'dst': '9', 'srcs': []}
{'morphs': [{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}, {'surface': 'コンピュータ', 'base': 'コンピュータ', 'pos': '名詞', 'pos1': '一般'}], 'dst': '8', 'srcs': []}
{'morphs': [{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'という', 'base': 'という', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '9', 'srcs': [7]}
{'morphs': [{'surface': '道具', 'base': '道具', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '10', 'srcs': [5, 6, 8]}
{'morphs': [{'surface': '用い', 'base': '用いる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}], 'dst': '12', 'srcs': [9]}
{'morphs': [{'surface': '『', 'base': '『', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}, {'surface': '』', 'base': '』', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '12', 'srcs': []}
{'morphs': [{'surface': '研究', 'base': '研究', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'する', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}], 'dst': '13', 'srcs': [10, 11]}
{'morphs': [{'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': '機', 'base': '機', 'pos': '名詞', 'pos1': '接尾'}, {'surface': '科学', 'base': '科学', 'pos': '名詞', 'pos1': '一般'}], 'dst': '14', 'srcs': [12]}
{'morphs': [{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}], 'dst': '15', 'srcs': [13]}
{'morphs': [{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}, {'surface': '分野', 'base': '分野', 'pos': '名詞', 'pos1': '一般'}, {'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '16', 'srcs': [14]}
{'morphs': [{'surface': '指す', 'base': '指す', 'pos': '動詞', 'pos1': '自立'}], 'dst': '17', 'srcs': [15]}
{'morphs': [{'surface': '語', 'base': '語', 'pos': '名詞', 'pos1': '一般'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], 'dst': '34', 'srcs': [0, 1, 3, 16]}
{'morphs': [{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '言語', 'base': '言語', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}], 'dst': '20', 'srcs': []}
{'morphs': [{'surface': '理解', 'base': '理解', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'や', 'base': 'や', 'pos': '助詞', 'pos1': '並立助詞'}], 'dst': '20', 'srcs': []}
{'morphs': [{'surface': '推論', 'base': '推論', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}], 'dst': '21', 'srcs': [18, 19]}
{'morphs': [{'surface': '問題', 'base': '問題', 'pos': '名詞', 'pos1': 'ナイ形容詞語幹'}, {'surface': '解決', 'base': '解決', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'など', 'base': 'など', 'pos': '助詞', 'pos1': '副助詞'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}], 'dst': '22', 'srcs': [20]}
{'morphs': [{'surface': '知的', 'base': '知的', 'pos': '名詞', 'pos1': '一般'}, {'surface': '行動', 'base': '行動', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'を', 'base': 'を', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '24', 'srcs': [21]}
{'morphs': [{'surface': '人間', 'base': '人間', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'に', 'base': 'に', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '24', 'srcs': []}
{'morphs': [{'surface': '代わっ', 'base': '代わる', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'て', 'base': 'て', 'pos': '助詞', 'pos1': '接続助詞'}], 'dst': '26', 'srcs': [22, 23]}
{'morphs': [{'surface': 'コンピューター', 'base': 'コンピューター', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'に', 'base': 'に', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '26', 'srcs': []}
{'morphs': [{'surface': '行わ', 'base': '行う', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'せる', 'base': 'せる', 'pos': '動詞', 'pos1': '接尾'}], 'dst': '27', 'srcs': [24, 25]}
{'morphs': [{'surface': '技術', 'base': '技術', 'pos': '名詞', 'pos1': '一般'}, {'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}, {'surface': 'または', 'base': 'または', 'pos': '接続詞', 'pos1': '*'}, {'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}], 'dst': '34', 'srcs': [26]}
{'morphs': [{'surface': '「', 'base': '「', 'pos': '記号', 'pos1': '括弧開'}, {'surface': '計算', 'base': '計算', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': '機', 'base': '機', 'pos': '名詞', 'pos1': '接尾'}], 'dst': '29', 'srcs': []}
{'morphs': [{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}, {'surface': 'コンピュータ', 'base': 'コンピュータ', 'pos': '名詞', 'pos1': '一般'}, {'surface': '）', 'base': '）', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'による', 'base': 'による', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '31', 'srcs': [28]}
{'morphs': [{'surface': '知的', 'base': '知的', 'pos': '名詞', 'pos1': '形容動詞語幹'}, {'surface': 'な', 'base': 'だ', 'pos': '助動詞', 'pos1': '*'}], 'dst': '31', 'srcs': []}
{'morphs': [{'surface': '情報処理', 'base': '情報処理', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'システム', 'base': 'システム', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'の', 'base': 'の', 'pos': '助詞', 'pos1': '連体化'}], 'dst': '33', 'srcs': [29, 30]}
{'morphs': [{'surface': '設計', 'base': '設計', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'や', 'base': 'や', 'pos': '助詞', 'pos1': '並立助詞'}], 'dst': '33', 'srcs': []}
{'morphs': [{'surface': '実現', 'base': '実現', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': 'に関する', 'base': 'に関する', 'pos': '助詞', 'pos1': '格助詞'}], 'dst': '34', 'srcs': [31, 32]}
{'morphs': [{'surface': '研究', 'base': '研究', 'pos': '名詞', 'pos1': 'サ変接続'}, {'surface': '分野', 'base': '分野', 'pos': '名詞', 'pos1': '一般'}, {'surface': '」', 'base': '」', 'pos': '記号', 'pos1': '括弧閉'}, {'surface': 'と', 'base': 'と', 'pos': '助詞', 'pos1': '格助詞'}, {'surface': 'も', 'base': 'も', 'pos': '助詞', 'pos1': '係助詞'}], 'dst': '35', 'srcs': [17, 27, 33]}
{'morphs': [{'surface': 'さ', 'base': 'する', 'pos': '動詞', 'pos1': '自立'}, {'surface': 'れる', 'base': 'れる', 'pos': '動詞', 'pos1': '接尾'}, {'surface': '。', 'base': '。', 'pos': '記号', 'pos1': '句点'}], 'dst': '-1', 'srcs': [34, 35]}
'''
