import part1.part1 as p1

def main():
    print('\n' + '[problem 00]')
    print(p1.reverse_str())
    #print(reverse_str('apple'))

    print('\n' + '[problem 01]')
    print(p1.extract_odd_char())

    print('\n' + '[problem 02]')
    print(p1.connect_str())

    print('\n' + '[problem 03]')
    print(p1.count_words())

    print('\n' + '[problem 04]')
    print(p1.disassemble_words())

    print('\n' + '[problem 05]')
    print(p1.make_ngram(True, 2, "I am an NLPer"))
    print(p1.make_ngram(False, 2, "I am an NLPer"))

    print('\n' + '[problem 06]')
    sum_set, mul_set, sub_set = p1.compare_bigram()
    print("和集合")
    print(sum_set)
    print('\n' + "積集合")
    print(mul_set)
    print('\n' + "差集合")
    print(sub_set)

    print('\n' + '[problem 07]')
    print(p1.make_template_sentence(12, '気温', 22.4))

    print('\n' + '[problem 08]')
    print(p1.cipher())


    print('\n' + '[problem 09]')
    print(p1.swap_char())

if __name__ == '__main__':
    main()
