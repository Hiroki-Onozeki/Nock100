import json
import re
from sklearn.cluster import KMeans
import gensim

def output_country_kmeans():
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

    kmean_model = KMeans(n_clusters=5)
    kmean_model.fit(vectors)

    # どのクラスタに分けられたか見てみる
    cluster_labels = kmean_model.labels_  
    cluster_country_list = [[],[],[],[],[]]
    for cluster_id, country_word in zip(cluster_labels, clean_country_set):
        cluster_country_list[cluster_id].append(country_word)
    print(cluster_country_list)
    print(len(clean_country_set))
    return


output_country_kmeans()

'''
実行結果

国名一覧　https://qiita.com/HirMtsd/items/6532ebfd7c486c3b7b44
185

['Togo', 'Gabon', 'Yemen', 'Malawi', 'Madagascar', 'Djibouti', 'Mauritania', 'Namibia', 'Guinea', 'Libya', 'Ethiopia', 'Egypt', 'Mali', 'Kenya', 'Liberia', 'Uganda', 'Senegal', 'Algeria', 'Burundi', 'Cameroon', 'Haiti', 'Nigeria', 'Benin', 'Gambia', 'Zambia', 'Lesotho', 'Botswana', 'Ghana', 'Rwanda', 'Sudan', 'Mozambique', 'Tunisia', 'Zimbabwe', 'Angola', 'Myanmar', 'Somalia', 'Eritrea', 'Comoros', 'Niger', 'Congo'], 
['Greenland', 'China', 'Luxembourg', 'France', 'Bangladesh', 'Chad', 'Territories', 'Germany', 'Ireland', 'Belgium', 'Mongolia', 'Malaysia', 'Philippines', 'Italy', 'Norway', 'Australia', 'Bhutan', 'Jersey', 'Iceland', 'Gibraltar', 'Austria', 'Iraq', 'Republic', 'Sweden', 'Singapore', 'Morocco', 'Finland', 'Jordan', 'Liechtenstein', 'Monaco', 'Israel', 'Bahrain', 'Denmark', 'Malta', 'Saba', 'Nepal', 'Iran', 'Netherlands', 'Man', 'Lebanon', 'Korea', 'Indonesia', 'Cambodia', 'Emirates', 'Macao', 'Kuwait', 'Pakistan', 'Oman', 'Qatar', 'Thailand', 'Japan', 'Taiwan', 'India', 'Antarctica', 'Canada', 'Switzerland', 'Afghanistan'], 
['Futuna', 'Tuvalu', 'Dominica', 'Réunion', 'Bahamas', 'Palau', 'Jamaica', 'Anguilla', 'Cocos', 'Vanuatu', 'Grenada', 'Samoa', 'Guadeloupe', 'Montserrat', 'Barbados', 'Fiji', 'Guam', 'Belize', 'Suriname', 'Bermuda', 'Pitcairn', 'Grenadines', 'Curaçao', 'Islands', 'Martinique', 'Mayotte', 'Tobago', 'Maldives', 'Niue', 'Kiribati', 'Tonga', 'Guyana', 'Guernsey', 'Tokelau', 'Seychelles', 'Micronesia', 'Barbuda', 'Aruba', 'Mauritius', 'Nauru'], 
['Herzegovina', 'Ukraine', 'Estonia', 'Hungary', 'Azerbaijan', 'Slovenia', 'Czechia', 'Latvia', 'Andorra', 'Belarus', 'Montenegro', 'Albania', 'Turkey', 'Kyrgyzstan', 'Turkmenistan', 'Romania', 'Greece', 'Uzbekistan', 'Croatia', 'Kazakhstan', 'Cyprus', 'Serbia', 'Lithuania', 'Moldova', 'Georgia', 'Bulgaria', 'Slovakia', 'Poland', 'Tajikistan', 'Armenia'], 
['Mexico', 'Bolivia', 'Cuba', 'Panama', 'Spain', 'Uruguay', 'Ecuador', 'Honduras', 'Peru', 'Nicaragua', 'Venezuela', 'Chile', 'Portugal', 'Brazil', 'Guatemala', 'Colombia', 'Paraguay', 'Argentina']
'''