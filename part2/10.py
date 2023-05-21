#ファイル内の行数をカウントする
def count_line_quantity(file_input = "popular-names.txt"):
    line_quantity = 0
    #mode='r' 読み込み専用はデフォルト
    with open(file_input) as f:
        for line in f:
            line_quantity += 1
    return line_quantity


print(count_line_quantity())

'''
実行結果
2780

wc popular-names.txt
2780   11120   55026 popular-names.txt
行、単語、バイト数
'''