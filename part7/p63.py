import gensim

# w1の単語ベクトルからw2のベクトルを引き，w3のベクトルを足したベクトルを計算し，類似度の高い10語
def output_vector_words(w1, w2, w3, n):
    '''
    これでもできたが良くないかも。w1~3が結果に出てきやすい
    v1 = p60.output_word_vector(w1)
    v2 = p60.output_word_vector(w2)
    v3 = p60.output_word_vector(w3)
    v = v1 - v2 + v3
    '''

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    top_simirality_words_list = model.most_similar(positive=[w1, w3], negative=[w2], topn=n)

    return top_simirality_words_list

'''
w_list = output_vector_words("Spain", "Madrid", "Athens", 10)
for e in w_list:
    print(e)
'''


'''
実行結果

('Greece', 0.6898480653762817)
('Aristeidis_Grigoriadis', 0.5606847405433655)
('Ioannis_Drymonakos', 0.5552908778190613)
('Greeks', 0.545068621635437)
('Ioannis_Christou', 0.5400862693786621)
('Hrysopiyi_Devetzi', 0.5248445272445679)
('Heraklio', 0.5207759737968445)
('Athens_Greece', 0.516880989074707)
('Lithuania', 0.5166866183280945)
('Iraklion', 0.5146791338920593)

変更前
('Athens', 0.7528454065322876)
('Greece', 0.6685471534729004)
('Aristeidis_Grigoriadis', 0.5495778322219849)
('Ioannis_Drymonakos', 0.5361457467079163)
('Greeks', 0.5351786613464355)
('Ioannis_Christou', 0.5330225825309753)
('Hrysopiyi_Devetzi', 0.5088489651679993)
('Iraklion', 0.5059264898300171)
('Greek', 0.5040615200996399)
('Athens_Greece', 0.5034109950065613)
'''