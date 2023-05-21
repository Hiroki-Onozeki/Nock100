# 引数で渡された自然数Nに対して、末尾からN行だけを出力
def output_lastnlines_from_file(file_input, n):
    contents_string = ""
    # 末尾n行をリストに格納
    with open(file_input) as f:
        contents_file_list = f.readlines()[-n:]

    # リストから取り出し文字列として変換
    for i in range(n):
        contents_string += contents_file_list[i]
    return contents_string


print(output_lastnlines_from_file("popular-names.txt", 5))

'''
実行結果
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

UNIX
tail -n 5 popular-names.txt
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

-n 行指定
'''