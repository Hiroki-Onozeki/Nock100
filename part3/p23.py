import re
import p20

# 記事中のセクション名とそのレベルを出力する
# =２以上がセクションで、多くなるほどレベルが高くなる


def extract_section_name_level(title_str):
    # 問題20の関数を使用して記事を取得
    text_contents = p20.get_text_from_json(title_str)
    text_sentences = text_contents.split('\n')

    for line in text_sentences:
        # セクション名に該当する文を抽出  ==セクション名==(=は２以上)
        pattern_section = re.search(r'==.+==', line)
        if pattern_section != None:
            section_sentence = pattern_section.group()
            # =を除いた名前と、(=の数/2)-1をレベルとして表示
            print(re.sub(r'=', '', section_sentence) + ' レベル' +
                  str(int((section_sentence.count('='))/2-1)))
    return


extract_section_name_level("イギリス")

'''
20 p.16 抽象的な名前よりも具体的な名前を使う

実行結果
国名 レベル1
歴史 レベル1
地理 レベル1
主要都市 レベル2
気候 レベル2
政治 レベル1
元首 レベル2
法 レベル2
内政 レベル2
地方行政区分 レベル2
外交・軍事 レベル2
経済 レベル1
鉱業 レベル2
農業 レベル2
貿易 レベル2
不動産 レベル2
エネルギー政策 レベル2
通貨 レベル2
企業 レベル2
通信 レベル3
交通 レベル1
道路 レベル2
鉄道 レベル2
海運 レベル2
航空 レベル2
科学技術 レベル1
国民 レベル1
言語 レベル2
宗教 レベル2
婚姻 レベル2
移住 レベル2
教育 レベル2
医療 レベル2
文化 レベル1
食文化 レベル2
文学 レベル2
哲学 レベル2
音楽 レベル2
ポピュラー音楽 レベル3
映画 レベル2
コメディ レベル2
国花 レベル2
世界遺産 レベル2
祝祭日 レベル2
スポーツ レベル2
サッカー レベル3
クリケット レベル3
競馬 レベル3
モータースポーツ レベル3
野球 レベル3
 カーリング  レベル3
 自転車競技  レベル3
脚注 レベル1
関連項目 レベル1
外部リンク レベル1
'''
