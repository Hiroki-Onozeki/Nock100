#   ファイル内の１列目の文字列の出現頻度順にソートする
def sort_str_frequency(file_input):
    contents_word_list = []
    str_list = []
    freq_str_dict = {}
    with open(file_input) as f:
        contents_file_input = f.read()

    # 読み込んだファイルの内容をtabで分割し,二次元のリストに格納
    contents_sentence_list = contents_file_input.splitlines()
    for i in range(len(contents_sentence_list)):
        contents_word_list.append(contents_sentence_list[i].split())

    # キーを文字列、値にその出現回数を入れる辞書を作成
    for i in range(len(contents_sentence_list)):
        if((contents_word_list[i][0] in freq_str_dict.keys()) == True):
            freq_str_dict[contents_word_list[i][0]] += 1
        else:
            freq_str_dict[contents_word_list[i][0]] = 1

    # 辞書をkeyにラムダ式を使用して辞書の値でソート
    freq_str_list = sorted(freq_str_dict.items(),
                           key=lambda x: x[1], reverse=True)
    for list in freq_str_list:
        str_list.append(list[0])
    return freq_str_list


print(sort_str_frequency("popular-names.txt"))

'''
実行結果
['James', 'William', 'John', 'Robert', 'Mary', 'Charles', 'Michael', 'Elizabeth', 'Joseph', 'Margaret', 'George', 'Thomas', 'David', 'Richard', 'Helen', 'Frank', 'Christopher', 'Anna', 'Edward', 'Ruth', 'Patricia', 'Matthew', 'Dorothy', 'Emma', 'Barbara', 'Daniel', 'Joshua', 'Sarah', 'Linda', 'Jennifer', 'Emily', 'Jessica', 'Jacob', 'Mildred', 'Betty', 'Susan', 'Henry', 'Ashley', 'Nancy', 'Andrew', 'Florence', 'Marie', 'Donald', 'Amanda', 'Samantha', 'Karen', 'Lisa', 'Melissa', 'Madison', 'Olivia', 'Stephanie', 'Abigail', 'Ethel', 'Sandra', 'Mark', 'Frances', 'Carol', 'Angela', 'Michelle', 'Heather', 'Ethan', 'Isabella', 'Shirley', 'Kimberly', 'Amy', 'Ava', 'Virginia', 'Deborah', 'Brian', 'Jason', 'Nicole', 'Hannah', 'Sophia', 'Minnie', 'Bertha', 'Donna', 'Cynthia', 'Alice', 'Doris', 'Ronald', 'Brittany', 'Nicholas', 'Mia', 'Noah', 'Joan', 'Debra', 'Tyler', 'Ida', 'Clara', 'Judith', 'Taylor', 'Alexis', 'Alexander', 'Mason', 'Harry', 'Sharon', 'Steven', 'Tammy', 'Brandon', 'Liam', 'Anthony', 'Annie', 'Gary', 'Jeffrey', 'Jayden', 'Charlotte', 'Lillian', 'Kathleen', 'Justin', 'Austin', 'Chloe', 'Benjamin', 'Evelyn', 'Megan', 'Aiden', 'Harper', 'Elijah', 'Bessie', 'Larry', 'Rebecca', 'Lauren', 'Amelia', 'Logan', 'Oliver', 'Walter', 'Carolyn', 'Pamela', 'Lori', 'Laura', 'Tracy', 'Julie', 'Scott', 'Kelly', 'Crystal', 'Rachel', 'Lucas']

UNIX
cut -f1 popular-names.txt | sort | uniq -c | sort -r | sed -n -e 1,10p
118 James
111 William
108 Robert
108 John
92 Mary
75 Charles
74 Michael
73 Elizabeth
70 Joseph
60 Margaret

cut -f1 タブ区切りの１フィールド目を取り出す
sort -r 逆順にソート
uniq -c 出現回数を表示
sed -n -e 1,10p １から１０行目まで
'''
