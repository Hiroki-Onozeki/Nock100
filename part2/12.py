# ファイル内の各行の１列目、２列目だけを抜き出したものをファイルに書き出す
def extract_colum(file_input="popular-names.txt"):
    contents_word_list = []
    with open(file_input) as f:
        contents_file_input = f.read()

    # 読み込んだファイルの内容をtabで分割し,二次元のリストに格納
    contents_sentence_list = contents_file_input.splitlines()
    for i in range(len(contents_sentence_list)):
        contents_word_list.append(contents_sentence_list[i].split())

    # 二次元のリストから対象の列だけをファイルに書き込む
    for i in range(len(contents_sentence_list)):
        # aモードは追加で末尾に書き込む
        with open("col1.txt", 'a') as writer:
            writer.write(contents_word_list[i][0] + "\n")

    for i in range(len(contents_sentence_list)):
        with open("col2.txt", 'a') as writer:
            writer.write(contents_word_list[i][1] + "\n")

    return 0


extract_colum()

'''
実行結果
col1.txtの中身は
Mary
Anna
Emma

col2.txt
F
F
F


UNIXでの実行結果
cut -f 1 popular-names.txt | sed -n -e 1,3p
Mary
Anna
Emma

cut -f 2  popular-names.txt | sed -n -e 1,3p
F
F
F

cut -f 1 各行の１つ目の要素を取り出す、要素はタブで分けられている
sed -n 1,3p １から３行目を出力
    -e 処理内容を指定
'''
