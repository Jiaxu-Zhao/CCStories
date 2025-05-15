#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
from stanfordcorenlp import StanfordCoreNLP

# try:
#     from tkinter import *
#     import tkinter as tk
#     from tkinter import filedialog
# except:
#     from Tkinter import *
#     import Tkinter as tk
#     from Tkinter import filedialog

def main():
    # root = tk.Tk()
    # root.geometry("700x700")
    # root.title('Chinese Text Analysis')
    # T = Text(root, height=4, width=100)
    # T.pack(side=LEFT, fill=Y)

    # Establish connection to Stanford parser
    nlp = StanfordCoreNLP('./stanford-corenlp-4.5.1', lang='zh')
    s = 0
    t = 0
    tc = 0
    line = 0
    a = [0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0]
    for sample in open('/data2/data/wxx/LLaMA-Factory/data/ccstories/filter/result_qwen3_0.6b.txt', 'r', encoding='utf-8'):
    #for i in range(1):
        #sample = 'result:当太阳刚刚探出紫红的脑袋时,溪边有一群麻雀,还有一群金花,喜欢听春莺歌,河边有一群燕子,和一群鹅。第1课时:一起做游戏第2课时:快乐的小海,快乐的大鸟第3课时:各种不同的小鸟第4课时:树叶的各种不同的颜色第5课时:树上不同的颜色第6课时:多一些耐心,少一些急躁第7课时:多一些用心,少一些懒惰第8课时:风雨,还有雨第9课时:山,美丽的山,美丽的云第10课时:小蝌蚪,小青蛙,小云彩,咱们去“云台山”第11课时:一叶知秋,秋之留痕第12课时:春之雪,春之留痕第13课时:夏天,我们好像黑眼睛第14课时:小溪边,春天的云第15课时:小荷才露尖尖角,蓓蕾初露第16课时'
        # post1 = sample.find('title:')
        # if post1 != -1:
        #     continue
        # sample = sample[22:]
        sample = sample.replace('\n','').replace('"','')
    # set up dictionary for word level lookups and level counts
        word_level_dict = build_word_level_dict()
        word_levels = [0,0,0,0,0,0,0]
        non_HSK_words = []

    # set up dictionary for character level lookups and level counts
        character_level_dict = build_character_level_dict()
        character_levels = [0,0,0,0,0,0,0]
        non_HSK_characters = []

    # get user input
    # sample = get_input()
    # sample = get_user_file(root)

    # tokenize Chinese input
        tokens = tokenize_input(sample, nlp)

    # get sentence lengths
        sentence_lengths = get_sentence_info(tokens)

    # remove punctuation from tokens
        tokens = remove_punctuation(tokens)

    # Group and count tokens based on their HSK level
        get_HSK_level_word_counts(tokens, word_level_dict, word_levels, non_HSK_words)
        total_tokens = len(tokens)

    # Get frequencies of words by HSK level
        word_frequencies = [0,0,0,0,0,0,0]
        get_HSK_level_word_frequencies(total_tokens, word_levels, word_frequencies)

    # Get minimum HSK level needed to read 90% of the words
        min_level_words = get_min_HSK_level_words(word_frequencies)

    # Group and count characters based on their HSK level
        sample = isolate_characters(sample)
        get_HSK_level_character_counts(sample, character_level_dict, character_levels, non_HSK_characters)

    # Get frequencies of characters by HSK level
        character_frequencies = [0,0,0,0,0,0,0]
        total_characters = len(sample)
        get_HSK_level_character_frequencies(total_characters, character_levels, character_frequencies)

    # Get minimum HSK level needed to read 90% of the characters
        min_level_characters = get_min_HSK_level_characters(character_frequencies)

    # Display results
    # display_results(sentence_lengths, total_tokens, word_levels, non_HSK_words, word_frequencies, min_level_words, total_characters, character_levels, character_frequencies, min_level_characters, non_HSK_characters, T)
        print(word_frequencies,character_frequencies)
        line += 1
        for i in range(7):
            a[i] +=word_frequencies[i]
            b[i] +=character_frequencies[i]
    print(a,b,line)
    #     if len(sentence_lengths)>0:
    #         for i in range(sentence_lengths):
    #             s+=sentence_lengths[i]
    #     t+=total_tokens
    #     tc+=total_characters
    #     line+=1
    #     print(sentence_lengths,total_tokens,total_characters)


    #print(word_frequencies, character_frequencies)

    # root.mainloop()
    #print(s, t, tc)

def build_word_level_dict():
    with open("levels_data/HSK_1-6_word_data.txt", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            word_level_dict = {rows[1]:rows[2] for rows in reader}
    file.close()
    return word_level_dict

def build_character_level_dict():
    with open("levels_data/HSK_1-6_character_data.txt", 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            character_level_dict = {rows[0]:rows[1] for rows in reader}
    file.close()
    return character_level_dict

def get_input():
    sample = input ('请输入测试语句: ')
    return sample

# def get_user_file(root):
#     # Close file browser after file is chosen
#     root.update()
#     root.filename = filedialog.askopenfilename(initialdir=os.getcwd() + '/sample_data/', title="Select file", filetypes=[("Text Files", "*.txt")])
#     with open(root.filename,'r',encoding='utf-8') as f:
#         sample = f.read()
#         return sample

def tokenize_input(sample, nlp):
    print('sample:'+sample)
    tokens = nlp.word_tokenize(sample)
    # close server connection
    nlp.close()
    return tokens

def get_sentence_info(tokens):
    count = 0
    punct = ['。', '？', '」', '！','……']
    sentence_lengths = []
    # remove non-terminal punctuation
    exclude = '，：；——（）【】；‘“”/「《》@、#¥%&*-=+～·\n\t\r'
    tokens = [x for x in tokens if x not in exclude]
    for token in tokens:
        if token not in punct:
            count += 1
        else:
            sentence_lengths.append(count)
            count = 0
    return sentence_lengths

# To-do: remove digits 0-9, while preserving Chinese numbers
def remove_punctuation(tokens):
    exclude = '，：；。——（）【】；‘“”/？「」《》！@、#¥%&*-=+～·'
    return [x for x in tokens if x not in exclude]

def isolate_characters(sample):
    exclude = ' ，：；。——（）【】；‘“”/？「」《》！@、#¥%&*-=+～·\n\t\r'
    return [x for x in sample if x not in exclude]

def get_HSK_level_word_counts(tokens, word_level_dict, word_levels, non_HSK_words):
    for token in tokens:
        if token in word_level_dict:
            word_levels[int(word_level_dict[token]) - 1] += 1
        else:
            word_levels[6] += 1
            non_HSK_words.append(token)

def get_HSK_level_word_frequencies(total_tokens, word_levels, word_frequencies):
    for level in word_levels:
        if total_tokens == 0:
            continue
        word_frequencies.append(round(level/total_tokens, 3))

def get_min_HSK_level_words(word_frequencies):
    sum = 0
    for i in range(0,5):
        sum += word_frequencies[i]
        if sum >= 0.9:
            return i+1
    return 6

def get_HSK_level_character_counts(sample, character_level_dict, character_levels, non_HSK_characters):
    for character in sample:
        if character in character_level_dict:
            character_levels[int(character_level_dict[character]) - 1] += 1
        else:
            character_levels[6] += 1
            non_HSK_characters.append(character)

def get_HSK_level_character_frequencies(total_characters, character_levels, character_frequencies):
    for level in character_levels:
        character_frequencies.append(round(level/total_characters,3))

def get_min_HSK_level_characters(character_frequencies):
    sum = 0
    for i in range(0,5):
        sum += character_frequencies[i]
        if sum >= 0.9:
            return i+1
    return 6

# def display_results(sentence_lengths, total_tokens, word_levels, non_HSK_words, word_frequencies, min_level_words, total_characters, character_levels, character_frequencies, min_level_characters, non_HSK_characters, T):
#
#     T.insert(END, '\nTotal number of words in text:        {}'.format(total_tokens))
#     T.insert(END, '\nTotal number of sentences in text:    {}'.format(len(sentence_lengths)))
#     T.insert(END, '\nAverage number of words per sentence: {}'.format(total_tokens / len(sentence_lengths)))
#     T.insert(END, '\nMinimum level to read 90% of words:      HSK {}'.format(min_level_words))
#     T.insert(END, '\nMinimum level to read 90% of characters: HSK {}'.format(min_level_characters))
#
#     T.insert(END, '\n\nPercentage of sentences with 0-8 words:   {:.1%}'.format(sum(i < 8 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 9-11 words:  {:.1%}'.format(sum(8 < i < 12 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 12-14 words: {:.1%}'.format(sum(11 < i < 15 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 15-16 words: {:.1%}'.format(sum(14 < i < 17 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 17-20 words: {:.1%}'.format(sum(16 < i < 21 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 21-24 words: {:.1%}'.format(sum(20 < i < 25 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 25-28 words: {:.1%}'.format(sum(24 < i < 29 for i in sentence_lengths) / len(sentence_lengths)))
#     T.insert(END, '\nPercentage of sentences with 29+ words:   {:.1%}'.format(sum(i > 28 for i in sentence_lengths) / len(sentence_lengths)))
#
#     T.insert(END, '\n\nNumber of HSK 6 words: {} Frequency: {:.1%}'.format(word_levels[5], word_frequencies[5]))
#     T.insert(END, '\nNumber of HSK 5 words: {} Frequency: {:.1%}'.format(word_levels[4], word_frequencies[4]))
#     T.insert(END, '\nNumber of HSK 4 words: {} Frequency: {:.1%}'.format(word_levels[3], word_frequencies[3]))
#     T.insert(END, '\nNumber of HSK 3 words: {} Frequency: {:.1%}'.format(word_levels[2], word_frequencies[2]))
#     T.insert(END, '\nNumber of HSK 2 words: {} Frequency: {:.1%}'.format(word_levels[1], word_frequencies[1]))
#     T.insert(END, '\nNumber of HSK 1 words: {} Frequency: {:.1%}'.format(word_levels[0], word_frequencies[0]))
#     T.insert(END, '\nNumber of words not found in HSK: {} Frequency: {:.1%}'.format(word_levels[6], word_frequencies[6]))
#     T.insert(END, '\n\nNon HSK words: {}'.format(non_HSK_words))
#
#     T.insert(END, '\n\nTotal number of characters in text:       {}'.format(total_characters))
#
#     T.insert(END, '\n\nNumber of HSK 6 characters: {} Frequency: {:.1%}'.format(character_levels[5], character_frequencies[5]))
#     T.insert(END, '\nNumber of HSK 5 characters: {} Frequency: {:.1%}'.format(character_levels[4], character_frequencies[4]))
#     T.insert(END, '\nNumber of HSK 4 characters: {} Frequency: {:.1%}'.format(character_levels[3], character_frequencies[3]))
#     T.insert(END, '\nNumber of HSK 3 characters: {} Frequency: {:.1%}'.format(character_levels[2], character_frequencies[2]))
#     T.insert(END, '\nNumber of HSK 2 characters: {} Frequency: {:.1%}'.format(character_levels[1], character_frequencies[1]))
#     T.insert(END, '\nNumber of HSK 1 characters: {} Frequency: {:.1%}'.format(character_levels[0], character_frequencies[0]))
#     T.insert(END, '\nNumber of characters not found in HSK: {} Frequency: {:.1%}'.format(character_levels[6], character_frequencies[6]))
#
#     T.insert(END, '\n\nNon HSK characters: {}'.format(non_HSK_characters))

if __name__ == '__main__':
    main()
