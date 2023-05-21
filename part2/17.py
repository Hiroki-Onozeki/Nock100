# 一列目の文字列のユニークな集合をソートしてリストとして出力する
def set_uniq_str(file_input):
    contents_word_list = []
    uniq_str_list = []
    with open(file_input) as f:
        contents_file_input = f.read()

    # 読み込んだファイルの内容をtabで分割し,二次元のリストに格納
    contents_sentence_list = contents_file_input.splitlines()
    for i in range(len(contents_sentence_list)):
        contents_word_list.append(contents_sentence_list[i].split())

    # リストに含まれていないなら追加する
    for i in range(len(contents_sentence_list)):
        if ((contents_word_list[i][0] in uniq_str_list) == False):
            uniq_str_list.append(contents_word_list[i][0])
    uniq_str_list.sort()
    return uniq_str_list


print(set_uniq_str("popular-names.txt"))

'''
実行結果
['Abigail', 'Aiden', 'Alexander', 'Alexis', 'Alice', 'Amanda', 'Amelia', 'Amy', 'Andrew', 'Angela', 'Anna', 'Annie', 'Anthony', 'Ashley', 'Austin', 'Ava', 'Barbara', 'Benjamin', 'Bertha', 'Bessie', 'Betty', 'Brandon', 'Brian', 'Brittany', 'Carol', 'Carolyn', 'Charles', 'Charlotte', 'Chloe', 'Christopher', 'Clara', 'Crystal', 'Cynthia', 'Daniel', 'David', 'Deborah', 'Debra', 'Donald', 'Donna', 'Doris', 'Dorothy', 'Edward', 'Elijah', 'Elizabeth', 'Emily', 'Emma', 'Ethan', 'Ethel', 'Evelyn', 'Florence', 'Frances', 'Frank', 'Gary', 'George', 'Hannah', 'Harper', 'Harry', 'Heather', 'Helen', 'Henry', 'Ida', 'Isabella', 'Jacob', 'James', 'Jason', 'Jayden', 'Jeffrey', 'Jennifer', 'Jessica', 'Joan', 'John', 'Joseph', 'Joshua', 'Judith', 'Julie', 'Justin', 'Karen', 'Kathleen', 'Kelly', 'Kimberly', 'Larry', 'Laura', 'Lauren', 'Liam', 'Lillian', 'Linda', 'Lisa', 'Logan', 'Lori', 'Lucas', 'Madison', 'Margaret', 'Marie', 'Mark', 'Mary', 'Mason', 'Matthew', 'Megan', 'Melissa', 'Mia', 'Michael', 'Michelle', 'Mildred', 'Minnie', 'Nancy', 'Nicholas', 'Nicole', 'Noah', 'Oliver', 'Olivia', 'Pamela', 'Patricia', 'Rachel', 'Rebecca', 'Richard', 'Robert', 'Ronald', 'Ruth', 'Samantha', 'Sandra', 'Sarah', 'Scott', 'Sharon', 'Shirley', 'Sophia', 'Stephanie', 'Steven', 'Susan', 'Tammy', 'Taylor', 'Thomas', 'Tracy', 'Tyler', 'Virginia', 'Walter', 'William']

UNIX
cut -f1 popular-names.txt | sort -u | sed -n -e 1,10p
Abigail
Aiden
Alexander
Alexis
Alice
Amanda
Amelia
Amy
Andrew
Angela

cut -f1 タブ区切りの１フィールド目を取り出す
sort -u 重複なしで並べ替え(= uniq | sort)
sed -n -e 1,10p １から１０行目まで
'''
