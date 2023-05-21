import pandas as pd


# 単語とidのファイルを作成する
def return_word_id(input_titles):
    word_id_dict = {}
    removed_word_id_dict = {}
    sorted_word_id_dict = {}
    id_list = []
    all_id_list = []
    train_df = pd.read_csv("train.txt", sep='\t')
    train_title = train_df["title"]

    # 単語と出現頻度の辞書を作る
    for title in train_title:
        word_list = title.split()
        for word in word_list:
            if word in word_id_dict:
                word_id_dict[word] += 1
            else: 
                word_id_dict[word] = 1
    
    # 頻度が１のものを削除
    for key, value in word_id_dict.items():
        if value != 1:
            removed_word_id_dict[key] = value
    
    # 辞書を値でソートして頻度順にidを付与する
    sorted_word_id_list = sorted(removed_word_id_dict.items(), key=lambda x:x[1], reverse=True)
    for i in range(len(sorted_word_id_list)):
        sorted_word_id_dict[sorted_word_id_list[i][0]] = i+1
    
    """
    # 単語列をidに変換し、列にしてかえす
    title_words = input_title.split()
    for word in title_words:
        if word in sorted_word_id_dict.keys():
            id_list.append(sorted_word_id_dict[word])
        else:
            id_list.append(0)
    return id_list

    """ 
    for title in input_titles:
        id_list = []
        title_words = title.split()
        for word in title_words:
            if word in sorted_word_id_dict.keys():
                id_list.append(int(sorted_word_id_dict[word]))
            else:
                id_list.append(0)
        all_id_list.append(id_list)
    return all_id_list

#print(return_word_id(["Amazon Plans to Fight FTC Over Mobile-App Purchases", "How to Convince a Loved One With Alzheimer's Symptoms to Go to the Doctor"]))


"""
実行結果

"How to Convince a Loved One With Alzheimer's Symptoms to Go to the Doctor"
[95, 1, 0, 18, 8940, 145, 19, 1605, 0, 1, 659, 1, 13, 2754]
"""

def return_word_id_dict():
    word_id_dict = {}
    removed_word_id_dict = {}
    sorted_word_id_dict = {}
    id_list = []
    train_df = pd.read_csv("train.txt", sep='\t')
    train_title = train_df["title"]

    # 単語と出現頻度の辞書を作る
    for title in train_title:
        word_list = title.split()
        for word in word_list:
            if word in word_id_dict:
                word_id_dict[word] += 1
            else: 
                word_id_dict[word] = 1
    
    # 頻度が１のものを削除
    for key, value in word_id_dict.items():
        if value != 1:
            removed_word_id_dict[key] = value

    # 辞書を値でソートして頻度順にidを付与する
    sorted_word_id_list = sorted(removed_word_id_dict.items(), key=lambda x:x[1], reverse=True)
    for i in range(len(sorted_word_id_list)):
        sorted_word_id_dict[sorted_word_id_list[i][0]] = i+1

    return sorted_word_id_dict

