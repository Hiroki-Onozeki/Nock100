import random
from operator import is_

# 問題0 文字列の逆順を出力


def reverse_str(reversing_msg="stressed"):
    reversed_msg = ''
    # 取得した文字を既存の文字列の先頭に格納していく
    for char in reversing_msg:
        reversed_msg = char + reversed_msg
    return reversed_msg

# 実行結果 desserts

# 問題1 文字列の奇数番目を抽出し連結したものを出力


def extract_odd_char(extracting_msg="パタトクカシーー"):
    extracted_msg = ''
    is_odd_num = True
    # 奇数ならば文字を抽出し順に格納していく
    for char in extracting_msg:
        if(is_odd_num == True):
            extracted_msg += char
            is_odd_num = False
        else:
            is_odd_num = True
    return extracted_msg

# 実行結果 パトカー

# 問題2 2つの文字列を交互に連結したものを出力


def connect_str(connecting_str1="パトカー", connecting_str2="タクシー"):
    connected_str = ''
    # 2つの文字列の長さが等しいと仮定する
    for char in range(len(connecting_str1)):
        connected_str += connecting_str1[char]
        connected_str += connecting_str2[char]
    return connected_str
# 実行結果 パタトクカシーー

# 問題3 文に含まれるアルファベットの文字数を出現順に並べたリストを出力


def count_words(sentence="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."):
    words_counter = {}
    # ディクショナリのキーに対象のアルファベットがなければ値を１として追加し、あればその値を+1する
    for char in sentence:
        if((char in words_counter) == True):
            words_counter[char] += 1
        else:
            words_counter[char] = 1
    return list(words_counter.values())
# 実行結果 [1, 6, 1, 14, 1, 6, 9, 2, 6, 4, 5, 1, 2, 4, 6, 4, 2, 4, 3, 4, 3, 1, 1, 1, 2, 1]

# 問題4 文内の単語の先頭文字とその単語の文内の位置を辞書型で出力


def disassemble_words(sentence="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."):
    words_counter = {}
    is_special = [1, 5, 6, 7, 8, 9, 15, 16, 19]
    words = sentence.split(' ')

    # 指定された番目の単語かどうかを判別し、適切な文字数を抽出し辞書に格納する
    for position_num in range(len(words)):
        if((position_num + 1 in is_special) == True):
            extracted_word = words[position_num]
            words_counter[extracted_word[0]] = position_num + 1
        else:
            extracted_word = words[position_num]
            words_counter[extracted_word[0] +
                          extracted_word[1]] = position_num + 1
    return words_counter
# 実行結果 {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mi': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20}


# 問題5 文から文字・単語n-gramを作成する


def make_ngram(is_word=True, n=2, sentence="I am an NLPer"):
    words = sentence.split(' ')
    ngram = []
    tmp = []

    # 文字と単語n-gramのどちらなのか判定し、ngramの要素を入れたリストを全体のリストに格納する
    if(is_word == True):
        for i in range(len(words) - n + 1):
            tmp.append(words[i])
            # 何個連続させるか
            for j in range(n-1):
                tmp.append(words[i + j + 1])
            # 書式変更
            ngram.append("\'" + ''.join(tmp) + "\'")
            tmp = []
        ngram = ','.join(ngram)
    else:
        sentence_except_blank = ''.join(words)
        for i in range(len(sentence_except_blank) - n + 1):
            tmp.append(sentence_except_blank[i])
            for j in range(n-1):
                tmp.append(sentence_except_blank[i + j + 1])
            ngram.append("\'" + ''.join(tmp) + "\'")
            tmp = []
        ngram = ','.join(ngram)

    return ngram


# 実行結果
'''
'Iam','aman','anNLPer'
'Ia','am','ma','an','nN','NL','LP','Pe','er'
'''


# 問題6 ２つの文字列の文字bi-gramuの和・差・積集合を求める


def compare_bigram(sentence1="paraparaparadise", sentence2="paragraph"):
    sum_set = []
    sub_set = []
    mul_set = []
    # 問題5の関数を使い、出力されたものをリスト形式にする
    # Xの名前を変更すべき
    X = make_ngram(False, 2, sentence1)
    X = X.replace('\'', '')
    X = X.split(',')
    Y = make_ngram(False, 2, sentence2)
    Y = Y.replace('\'', '')
    Y = Y.split(',')

    for elem in X:
        if((elem in sum_set) == False):
            sum_set.append(elem)
    for elem in Y:
        if((elem in sum_set) == False):
            sum_set.append(elem)

    #差集合は第一引数 - 第二引数とする
    for elem in X:
        if((elem in Y) == True):
            mul_set.append(elem)
        else:
            sub_set.append(elem)
    return sum_set, mul_set, sub_set


# 実行結果
'''
和集合
['pa', 'ar', 'ra', 'ap', 'ad', 'di', 'is', 'se', 'ag', 'gr', 'ph']

積集合
['pa', 'ar', 'ra', 'ap', 'pa', 'ar', 'ra', 'ap', 'pa', 'ar', 'ra']

差集合
['ad', 'di', 'is', 'se']
'''

# 問題7 引数x,y,z よりx時のyはzという文字列を出力


def make_template_sentence(x=12, y="気温", z=22.4):
    # x,y,z が文字列でないなら文字列に変換して出力
    # きれいな形ではない
    if(x != str):
        x = str(x)
    if(y != str):
        y = str(y)
    if(z != str):
        z = str(z)

    sentence = x + "時の" + y + "は" + z
    return sentence

# 実行結果 12時の気温は22.4

# 問題8 文字列の各文字を英小文字ならasc2に変換して出力


def cipher(msg='98Napです。'):
    encrypted_str = ''
    # 文字が英字かつ小文字かどうか判定し、asc2に変換
    for char in msg:
        if((char.islower() == True) and (char.isalpha() == True)):
            encrypted_str += str(ord(char))
        else:
            encrypted_str += char

    return encrypted_str

# 実行結果 98N97112です。

# 問題9 文中の各単語の先頭と末尾の文字は残し、それ以外をランダムに並び替えて出力


def swap_char(sentence="I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."):
    words = sentence.split(' ')
    shuffled_sentence_list = []

    # 単語の長さが5以上なら、先頭と末尾を一時的に格納し、それ以外をランダムに並び替える
    for i in words:
        if(len(i) <= 4):
            shuffled_sentence_list.append(i)
        else:
            top_char = i[0]
            last_char = i[len(i)-1]
            central_str = i[1:len(i)-1]
            # randomを使うために一度リストにして、シャッフル後に文字列に戻す
            random_str = ''.join(random.sample(central_str, len(central_str)))
            shuffled_sentence_list.append(top_char + random_str + last_char)
    return ' '.join(shuffled_sentence_list)

# 実行結果 I c'uolndt bvleiee that I cuold aacullty uaetnsdnrd what I was raenidg : the pnoemanehl poewr of the haumn mind .
