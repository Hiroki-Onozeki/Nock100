#
def replace_tab_to_sapce(file_input = "popular-names.txt"):
    #ファイルの内容を全て読み込む　；readは全部、readlineは１行,readlinesは全てを１行ごとのリストに
    with open(file_input) as f:
        contents_input_file = f.read()
    #タブ文字を認識しなかったので、エスケープシーケンスを使用
    contents_input_file = contents_input_file.replace('\t', ' ')

    #新たなファイルに内容を書き込む
    with open(file_input + "_sapace_replaced.txt", 'w') as writer: 
        writer.write(contents_input_file)
    return 0

replace_tab_to_sapce()

'''
実行結果
expand popular-names.txt | sed -n -e 1,3p
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880

expand popular-names.txt_sapace_replaced.txt | sed -n -e 1,3p 
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880

expand タブ区切りの入力に対してのみ、各列のデータをそろえて表示する
'''
