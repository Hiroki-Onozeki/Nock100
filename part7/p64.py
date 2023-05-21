import gensim

#　類似度の高い単語と類似度を追加したファイルを作成する
def make_top_similarity_file(input_file_name, output_file_name):
    with open(input_file_name) as f:
        for line in f:
            country_list = line.split()
            model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            
            # 類似度が最も高い単語と類似度を得る
            if len(country_list) >= 3:
                top_sim_list = model.most_similar(positive=[country_list[1], country_list[2]], negative=[country_list[0]], topn=1)
                country_list.append(top_sim_list[0][0])
                country_list.append(str(top_sim_list[0][1]))
                with open(output_file_name, 'a') as w:
                    w.write(" ".join(country_list) + '\n')
            else:
                with open(output_file_name, 'a') as w:
                    w.write(line)
    return


make_top_similarity_file("questions-words.txt", "questions-words-added.txt")


'''
実行結果

: capital-common-countries
Athens Greece Baghdad Iraq Iraqi 0.635187029838562
Athens Greece Bangkok Thailand Thailand 0.7137669324874878
Athens Greece Beijing China China 0.7235778570175171
Athens Greece Berlin Germany Germany 0.6734622716903687
Athens Greece Bern Switzerland Switzerland 0.4919748306274414
Athens Greece Cairo Egypt Egypt 0.7527808547019958
Athens Greece Canberra Australia Australia 0.583732545375824
Athens Greece Hanoi Vietnam Viet_Nam 0.6276341676712036
Athens Greece Havana Cuba Cuba 0.6460990905761719
'''