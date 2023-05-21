from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import re
import gensim
import json
import numpy as np


def output_tsne():
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

    # tsneを使って次元削減をし、二次元にする
    tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
    embedded = tsne.fit_transform(vectors)

    # 可視化する
    plt.figure(figsize=(10, 10))
    plt.scatter(np.array(embedded).T[0], np.array(embedded).T[1])
    for (x, y), name in zip(embedded, clean_country_set):
        plt.annotate(name, (x, y))
    plt.show()
    return

output_tsne()