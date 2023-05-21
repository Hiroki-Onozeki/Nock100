# 引数で渡された自然数Nに対して、行単位でファイルをN分割する

def divide_nfile(file_input, n):
    sum_lines = 0
    division_num_index = [0]
    # ファイルの行数を数える
    with open(file_input) as f:
        for line in f:
            sum_lines += 1

    # 分割する行目をリストに格納
    for i in range(n-1):
        division_num_index.append(int((sum_lines/n)*(i+1)))
    division_num_index.append(sum_lines)

    # 分割行目を基準にファイルを分割し、新たなファイルに格納
    for i in range(len(division_num_index)-1):
        contents_output = ""
        with open(file_input) as f:
            contents_file_input = f.readlines()[division_num_index[i]:division_num_index[i+1]]
        # リストを文字列に変換してからファイルに書き出す
        for j in range(len(contents_file_input)):
            contents_output += contents_file_input[j]
        with open("s" + str(i+1) + file_input, 'w') as writer:
            writer.write(contents_output)

    return 0


divide_nfile("popular-names.txt", 3)

'''
実行結果
s1popular-names.txt 926行
s2popular-names.txt 927行
s3popular-names.txt 927行
計 2780行

UNIX
split -n l/3
行単位で３個にファイルを分割するコマンド
-n:n個に -l:行数指定

macosのsplitとGNUのsplitコマンドは別であるため、-n,lオプションが使えなかった
新たにcoreutilsをインストールすればできるらしいが...
'''
