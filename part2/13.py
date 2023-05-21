# 二つのファイルの内容をtab区切りでを結合する
def marge_txt(file_input1="col1.txt", file_input2="col2.txt"):
    # ファイルの内容を取り出す
    with open(file_input1) as f1:
        contents_file1 = f1.read()
    with open(file_input2) as f2:
        contents_file2 = f2.read()

    # 取り出した内容を単語に分けてリストに格納
    word_list1 = contents_file1.split()
    word_list2 = contents_file2.split()

    # リストから順に取り出し、末尾に追加するaモードで書き込む
    for i in range(len(word_list1)):
        with open("marge-col1-col2.txt", 'a') as writer:
            writer.write(word_list1[i] + "\t")
            writer.write(word_list2[i] + "\n")
    return 0


marge_txt()


'''
実行結果
marge-col1-col2.txt の上から３行
Mary	F
Anna	F
Emma	F


UNIX
paste col1.txt 
col2.txt | sed -n -e 1,3p
Mary    F
Anna    F
Emma    F
'''
