import p60
import numpy as np

# ２つの単語のコサイン類似度を出力する
def output_cosine_similarity(word_str_1, word_str_2):
    word_vector_1 = p60.output_word_vector(word_str_1)
    word_vector_2 = p60.output_word_vector(word_str_2)

    # normでベクトルのノルムを得る
    cosine_similarity = np.dot(word_vector_1, word_vector_2)/(np.linalg.norm(word_vector_1)*np.linalg.norm(word_vector_2))
    return cosine_similarity

#print(output_cosine_similarity("United_States", "U.S."))


'''
実行結果

0.7310775 

ちなみにapple　-0.026893016
'''