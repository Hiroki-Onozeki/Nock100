import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import re
import gensim
import json



def output_country_ward():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    country_set = set()
    clean_country_set = set()
    vectors = []

    # 国名一覧ファイルを取得する
    json_open = open("Country_list.json")
    json_load = json.load(json_open)

    for i in range(len(json_load["countries"])):
        if "short" in json_load["countries"][i]["en_name"].keys():
            country_set.add(json_load["countries"][i]["en_name"]["short"])

    # 不要箇所を削除する
    for elem in country_set:
        elem = re.sub(r" \(.+\)", "", elem)
        elem = re.sub(r" \[.+\]", "", elem)
        elem = re.sub(r"\S+\s\S+", "", elem)
        elem = re.sub(r"[^-]+[-]+.+", "", elem)
        elem = re.sub(r"\s", "", elem)
        if elem != "" and elem != "Eswatini":
            clean_country_set.add(elem)

    # 国名をベクトルに変換し、クラスタリングを行う
    for country in clean_country_set:
        vectors.append(model[country])

    # ユークリッド距離、
    linkage_result = linkage(vectors, method='ward', metric='euclidean')
    plt.figure()
    dendrogram(linkage_result, labels=list(clean_country_set))
    plt.show()
    return

output_country_ward()

'''
実行結果

p68_ward.png
'''