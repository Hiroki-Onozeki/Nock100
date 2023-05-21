# ファイル内の３コラム目の数値の逆順にソートしたファイルを新たに作成する

def sort_int_in_file(file_input):
    contents_word_list = []
    with open(file_input) as f:
        contents_str_file = f.read()

    # 読み込んだファイルの内容をtabで分割し,二次元のリストに格納
    contents_sentence_list = contents_str_file.splitlines()
    for i in range(len(contents_sentence_list)):
        contents_word_list.append(contents_sentence_list[i].split())

    #　keyにラムダ式を使用して2列目の値でソート
    contents_word_list.sort(reverse=True, key=lambda x: x[2])

    # リストを文字列に変換してファイルに書き出す
    for i in range(len(contents_sentence_list)):
        with open("rev-sort-popular-names.txt", 'a') as writer:
            contents_sentence = '\t'.join(contents_word_list[i])
            writer.write(contents_sentence + '\n')
    return 0


sort_int_in_file("popular-names.txt")

'''
実行結果
rev-sort-popular-names.txt
Linda	F	99689	1947
James	M	9951	1911
Mildred	F	9921	1913
Mary	F	9889	1886
Mary	F	9888	1887

UNIX
sort -r -k 3 popular-names.txt | sed -n -e 1,5p
Linda   F       99689   1947
James   M       9951    1911
Mildred F       9921    1913
Mary    F       9889    1886
Mary    F       9888    1887

sort -r -k 逆順で３列目の値でソート
'''
