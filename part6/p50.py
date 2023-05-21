import pandas as pd
#from sklearn.model_selection import train_test_split


# 指定された情報源の記事のみのファイルを作成する
def make_train_valid_test_file(input_file_name="newsCorpora.csv"):
    col_name = ["id", "title", "url", "publisher", "category", "story", "hostname", "timestamp"]
    df = pd.read_csv(input_file_name, names=col_name, sep='\t')
    
    # 特定の情報源からのもののみを抽出する
    mask = (df["publisher"] == "Reuters") | (df["publisher"] == "Huffington Post") | (df["publisher"] == "Businessweek") | (df["publisher"] == "Contactmusic.com") | (df["publisher"] == "Daily Mail")
    df_special_publisher = df[mask]
    print(df_special_publisher.isnull().any())

    #  ランダムに並び替える fracは抽出する行の割合(１で１００パー) ignoreでインデックスを０からにする
    df_random = df_special_publisher.sample(frac=1, ignore_index=True)
    df_random.to_csv("all.txt", sep='\t', index=False)
    
    # ３つに分割してファイルに保存する　すでにシャッフルしたのでFalseにする 行インデックスも邪魔なので消しちゃう
    train_df, test_valid_df = train_test_split(df_random, test_size=0.2, shuffle=False)
    valid_df, test_df = train_test_split(test_valid_df, test_size=0.5, shuffle=False)
    train_df.to_csv("train.txt", sep='\t', index=False)
    valid_df.to_csv("valid.txt", sep='\t', index=False)
    test_df.to_csv("test.txt", sep='\t', index=False)

    print(train_df["category"].value_counts())
    print(valid_df["category"].value_counts())
    print(test_df["category"].value_counts())
    return


make_train_valid_test_file()


'''
実行結果

e    4105
b    3082
t     945
m     624
Name: category, dtype: int64
e    540
b    367
t    115
m     72
Name: category, dtype: int64
e    547
b    356
t    111
m     81
Name: category, dtype: int64
'''