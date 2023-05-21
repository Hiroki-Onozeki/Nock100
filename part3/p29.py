import p25
import requests

# 画像のURLを取得する


def return_img_url(title_str):
    # 問題２５の関数を実行して、基礎情報のテンプレートのフィールド名と値の辞書を得る
    basic_inf_dict = p25.extract_basicinf_as_dict(title_str)

    # ヒントのsample codeをそのまま持ってきた
    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": "File:" + basic_inf_dict["国旗画像"],
        # iipropのパラメータをurlにすれば、urlの部分がリターン結果に追加される
        "iiprop": "url"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    PAGES = DATA["query"]["pages"]
    for k, v in PAGES.items():
        # vの中身を見てurlがある部分だけを持ってくる
        print(v['imageinfo'][0]['url'])
    return


return_img_url("イギリス")

'''
28 p.51 コードを段落に分割する

実行結果
https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg
'''
