from pandas import read_csv
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import nltk
import sys
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Main:
    def __init__(self):
        pass
    
    def get_best_synset_pair(self, word1, word2):
        synsets_word1 = wn.synsets(word1)
        synsets_word2 = wn.synsets(word2)

        if len(synsets_word1) == 0 or len(synsets_word2) == 0:
            return None, None

        max_sim = -1.0
        best_synsets = None

        for synset1 in synsets_word1:
            for synset2 in synsets_word2:
                similarity = wn.path_similarity(synset1, synset2)
                print(f"{synset1}, {synset2} {similarity}")
                if similarity > max_sim:
                    max_sim = similarity
                    best_synsets = (synset1, synset2)

        return best_synsets



if __name__ == "__main__":
    main = Main()
    print(f"BEST PAIR: {main.get_best_synset_pair('RAM', 'memory')}")



