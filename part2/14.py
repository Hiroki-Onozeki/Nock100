# 引数で渡された自然数Nに対して、先頭からN行だけを出力
def output_headnlines_from_file(file_input, n):
    contents_file_input = ""
    # n回１行を抜き出すことを繰り返す
    with open(file_input) as f:
        for line_num in range(n):
            contents_file_input += f.readline()
    return contents_file_input


print(output_headnlines_from_file("popular-names.txt", 5))


'''
実行結果
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880

UNIX
head -n 5 popul
ar-names.txt
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880

-n 行指定
'''
