import gzip
import json

# Wikipedia記事のjsonファイルを読み込み、記事本文を表示する



def get_text_from_json(title_str):
    # バイナリファイルである？から、バイナリモードを使用してgzipの１行目をjsoとして読み込む
    with gzip.open("jawiki-country.json.gz", 'rb') as gzip_f:
        # 1行ごとがjson形式になっているため、文字列として抜き出しloadsで読み込む(loadはファイルをjsonに)
        for line in gzip_f:
            contents_dict = json.loads(line)
            if contents_dict["title"] == title_str:
                return contents_dict["text"]
    return


print(get_text_from_json("イギリス"))

'''
23 p.21 値の属性を追加する

実行結果
{{redirect|UK}}
{{redirect|英国|春秋時代の諸侯国|英 (春秋)}}
{{Otheruses|ヨーロッパの国|長崎県・熊本県の郷土料理|いぎりす}}
{{基礎情報 国
|略名  =イギリス
|日本語国名 = グレートブリテン及び北アイルランド連合王国
|公式国名 = {{lang|en|United Kingdom of Great Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />
*{{lang|gd|An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath}}（[[スコットランド・ゲール語]]）
*{{lang|cy|Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon}}（[[ウェールズ語]]）
*{{lang|ga|Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann}}（[[アイルランド語]]）
*{{lang|k
'''
