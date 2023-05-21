import re
import p20

# カテゴリ名のみを抽出;長くなるから国名(イギリス)指定する


def extract_category_name(title_str):
    # 問題20の関数を使用して記事を取得
    text_contents = p20.get_text_from_json(title_str)
    text_sentences = text_contents.split('\n')

    for line in text_sentences:
        # [[Category:イギリス|*]]　から不要箇所を削除
        pattern_category = re.search(r'.+Category:.+', line)

        if pattern_category != None:
            # subで置換して不要箇所を置換する。かなり強引なやり方
            result = re.sub(r'\[|\]|\|\*|Category:', '',
                            pattern_category.group())
            print(result)
    return


extract_category_name("イギリス")

'''
26 P.67 要約コメント

実行結果
イギリス
イギリス連邦加盟国
英連邦王国
G8加盟国
欧州連合加盟国|元
海洋国家
現存する君主国
島国
1801年に成立した国家・領域
'''
