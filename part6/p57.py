import pandas as pd
import pickle

# 特徴量の重みが高いものと低いものを出力する
def output_high_low_weight_feature():
    # 特徴量の重みと、特徴量に対応する単語を取得し、データフレームを作成する
    lr_model = pickle.load(open("lr_model.sav", 'rb'))
    feature_weight_list = lr_model.coef_
    test_feature_df = pd.read_csv("test.feature.txt", sep='\t')
    df = pd.DataFrame(feature_weight_list, columns=test_feature_df.columns, index=['row0', 'row1', 'row2', 'row3'])

    # それぞれのクラスに対してシリーズを作成し、ソートして重みが高いものと低いものを出力する
    b_df = df.loc['row0']
    b_df = b_df.sort_values(ascending=False)
    e_df = df.loc['row1']
    e_df = e_df.sort_values(ascending=False)
    m_df = df.loc['row2']
    m_df = m_df.sort_values(ascending=False)
    t_df = df.loc['row3']
    t_df = t_df.sort_values(ascending=False)

    for elem in b_df.index[:10]:
        print(elem, end=' ')
    print('\n')
    for elem in b_df.index[-10:]:
        print(elem, end=' ')
    print('\n')
    for elem in e_df.index[:10]:
        print(elem, end=' ')
    print('\n')
    for elem in e_df.index[-10:]:
        print(elem, end=' ')
    print('\n')
    for elem in m_df.index[:10]:
        print(elem, end=' ')
    print('\n')
    for elem in m_df.index[-10:]:
        print(elem, end=' ')
    print('\n')
    for elem in t_df.index[:10]:
        print(elem, end=' ')
    print('\n')
    for elem in t_df.index[-10:]:
        print(elem, end=' ')
    print('\n')
    return

output_high_low_weight_feature()


'''
b:business
高　us update stocks euro forex dollar china wall ecb on 
低　are with you google study of ebola is and the 

e:entertainment
高　kardashian kim the and her cyrus miley chris star kanye 
低　may says stocks could ebola google study to us update 

m:health
高　ebola study drug cancer mers fda could health risk are 
低　gm climate to euro kardashian on facebook apple stocks google 

t:science and technology
高　google facebook apple climate to gm you microsoft neutrality nasa 
低　study euro fda kardashian mers as cancer drug stocks ebola
'''