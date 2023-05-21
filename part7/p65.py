import re

# 意味的・文法的アナロジーの正解率を求める
def calculate_accuracy():
    is_gram = False
    sum_semantic_num = 0
    correct_semantic_num = 0
    sum_syntactic_num = 0
    correct_syntactic_num = 0

    with open("questions-words-added.txt") as f:
        for line in f:
            country_list = line.split()

            # カテゴリ名なら、文法的アナロジーへの分岐点を探し、フラグを立てる
            if len(country_list) < 4:
                gram_pattren = re.match(r'gram.+', country_list[1])
                if gram_pattren != None:
                    is_gram = True
            
            # 意味的・文法的を判断し、合計個数と正解個数を加算していく
            else:
                if is_gram == False:
                    sum_semantic_num += 1
                    if country_list[3] == country_list[4]:
                        correct_semantic_num += 1
                else:
                    sum_syntactic_num += 1
                    if country_list[3] == country_list[4]:
                        correct_syntactic_num += 1
            
    print(correct_semantic_num/sum_semantic_num)
    print(correct_syntactic_num/sum_syntactic_num)
    return

calculate_accuracy()


'''
実行結果

意味的アナロジー　0.7308602999210734
文法的アナロジー　0.7400468384074942
'''