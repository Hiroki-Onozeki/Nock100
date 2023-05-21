from graphviz import Digraph
import p41

# 文の係り受け木の有向グラフを作成する
def make_digraph_from_sentence(sentence_chunk):
    dg = Digraph(format='png')

    for elem_chunk in sentence_chunk:
        from_words = ""
        to_words = ""
        index_num = elem_chunk["dst"]

        # 係り元の単語を抽出
        for elem in elem_chunk["morphs"]:
                if elem["pos"] != "記号":
                    from_words += elem["surface"]

        # 係り先の単語を抽出
        for elem in sentence_chunk[int(index_num)]["morphs"]:
                if elem["pos"] != "記号":
                    to_words += elem["surface"]

        # エッジを貼る。ノードがない時は自動で作成される
        dg.edge(from_words, to_words)
    dg.render("./digraph", view=True)
    dg.view
    return

all_sentence_list = p41.input_txt_to_chunk()
sentence = all_sentence_list[1]
make_digraph_from_sentence(sentence)

'''
p.63 質問されそうなことを想像する

実行結果

digraph.png　を参照
'''