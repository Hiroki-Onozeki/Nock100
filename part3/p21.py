import re
import p20

# カテゴリ名を宣言している行を抽出;長くなるから国名指定する


def extract_category_sentence(title_str):
    # 問題20の関数を使用して記事を取得
    text_contents = p20.get_text_from_json(title_str)
    text_sentences = text_contents.split('\n')

    for line in text_sentences:
        # matchは先頭,searchは先頭に限らず：.任意の文字、+１回以上の繰り返し
        # rによってraw文字列になり、\の複雑な挙動(\\)を回避できるため、正規表現使用時にはr使用が推奨されている。今回はなくても動作は同じ
        pattern_category = re.search(r'.+Category:.+', line)
        if pattern_category != None:
            # group[0]マッチオブジェクトから文字列を抽出
            print(pattern_category.group())
    return


extract_category_sentence("イギリス")


'''
23 p.21 値の属性を追加する

実行結果
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]

マッチオブジェクトインスタンス：マッチした文字列、開始位置、終了位置などの情報を含む
group(0) マッチした文字列全体を返す
    1以上；対応する部分文字列
    引数２つ；タプルを返す
'''
