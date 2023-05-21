import gensim

# 類似度の高い１０個の語を出力する
def output_top_simirality_10words(input_word_str):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    top_simirality_words_list = model.most_similar(input_word_str)
    return top_simirality_words_list[:10]

'''
w_list = output_top_simirality_10words("United_States")
for e in w_list:
    print(e)
'''


'''
実行結果

('Unites_States', 0.7877248525619507)
('Untied_States', 0.7541370987892151)
('United_Sates', 0.7400725483894348)
('U.S.', 0.7310774922370911)
('theUnited_States', 0.6404394507408142)
('America', 0.6178409457206726)
('UnitedStates', 0.6167312264442444)
('Europe', 0.6132988929748535)
('countries', 0.6044804453849792)
('Canada', 0.6019068956375122)
'''