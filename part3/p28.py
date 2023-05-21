import p27
import re

# 他のマークアップを削除する


def delete_other_mark(title_str):
    clean_basic_inf_dict = {}
    # 問題２７の関数を実行して、基礎情報から強調・内部リンクマークアップを除去した辞書を得る
    basic_inf_dict = p27.delete_inside_link_mark(title_str)

    # 取り除きたい表現と一致したら、除去する
    for key, value in basic_inf_dict.items():
        value = re.sub(r'\[\[ファイル:.*\||\]\]', '', value)
        value = re.sub(r'ファイル:.*', '', value)
        value = re.sub(r'\[\[Category:.*\||\]\]', '', value)
        value = re.sub(r'\[http.*\]', '', value)
        value = re.sub(r'http.*', '', value)
        value = re.sub(r'\<[^>]+\>', '', value)
        value = re.sub(r'\{\{[^}]*\|([^}]*)\}\}', r'\1', value)
        value = re.sub(r'\{\{[^}]+\}\}', '', value)
        value = re.sub(r'\{\{.*|\}\}', '', value)

        # 一部キーにも含まれているマークアップも同様に除去する
        key = re.sub(r'\[\[ファイル:.*\||\]\]', '', key)
        key = re.sub(r'\[\[Category:.*\||\]\]', '', key)
        key = re.sub(r'\[http.*\]', '', key)
        key = re.sub(r'http.*', '', key)
        key = re.sub(r'\<.*\>', '', key)

        clean_basic_inf_dict[key] = value
    return clean_basic_inf_dict


output_dict = delete_other_mark("イギリス")
for item in output_dict.items():
    print(item)


'''
28 p.51 コードを段落に分割する

実行結果
('略名', 'イギリス')
('日本語国名', 'グレートブリテン及び北アイルランド連合王国')
('公式国名', 'United Kingdom of Great Britain and Northern Ireland英語以外での正式国名:\n*An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath（スコットランド・ゲール語）\n*Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon（ウェールズ語）\n*Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann（アイルランド語）\n*An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh（コーンウォール語）\n*Unitit Kinrick o Great Breetain an Northren Ireland（スコットランド語）\n**Claught Kängrick o Docht Brätain an Norlin Airlann、Unitet Kängdom o Great Brittain an Norlin Airlann（アルスター・スコットランド語）')
('国旗画像', 'Flag of the United Kingdom.svg')
('国章画像', 'イギリスの国章')
('国章リンク', '（国章）')
('標語', 'Dieu et mon droit（フランス語:神と我が権利）')
('国歌', 'God Save the Queen神よ女王を護り賜え')
('地図画像', 'Europe-UK.svg')
('位置画像', 'United Kingdom (+overseas territories) in the World (+Antarctica claims).svg')
('公用語', '英語')
('首都', 'ロンドン（事実上）')
('最大都市', 'ロンドン')
('元首等肩書', '女王')
('元首等氏名', 'エリザベス2世')
('首相等肩書', '首相')
('首相等氏名', 'ボリス・ジョンソン')
('他元首等肩書1', '貴族院議長')
('他元首等氏名1', 'ノーマン・ファウラー')
('他元首等肩書2', '庶民院議長')
('他元首等氏名2', 'Lindsay Hoyle')
('他元首等肩書3', '最高裁判所長官')
('他元首等氏名3', 'ブレンダ・ヘイル')
('面積順位', '76')
('面積大きさ', '1 E11')
('面積値', '244,820')
('水面積率', '1.3%')
('人口統計年', '2018')
('人口順位', '22')
('人口大きさ', '1 E7')
('人口値 6643万5600{{Cite weburl=', '6643万5600')
('人口密度値', '271')
('GDP統計年元', '2012')
('GDP値元 1兆5478億[', '1兆5478億')
('GDP統計年MER', '2012')
('GDP順位MER', '6')
('GDP値MER 2兆4337億<ref name=', '2兆4337億')
('GDP統計年', '2012')
('GDP順位', '6')
('GDP値 2兆3162億<ref name=', '2兆3162億')
('GDP/人 36,727<ref name=', '36,727')
('建国形態', '建国')
('確立形態1', 'イングランド王国／スコットランド王国（両国とも1707年合同法まで）')
('確立年月日1', '927年／843年')
('確立形態2', 'グレートブリテン王国成立（1707年合同法）')
('確立年月日2', '1707年5月1日')
('確立形態3', 'グレートブリテン及びアイルランド連合王国成立（1800年合同法）')
('確立年月日3', '1801年1月1日')
('確立形態4', '現在の国号「グレートブリテン及び北アイルランド連合王国」に変更')
('確立年月日4', '1927年4月12日')
('通貨', 'UKポンド (£)')
('通貨コード', 'GBP')
('時間帯', '±0')
('夏時間', '+1')
('ISO 3166-1', 'GB / GBR')
('ccTLD', '.uk / .gb使用は.ukに比べ圧倒的少数。')
('国際電話番号', '44')
('注記', '\n\n')
'''
