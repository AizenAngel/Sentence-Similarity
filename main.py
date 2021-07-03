from os import fsync
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
        self.ALPHA = 0.2
        self.BETA = 0.45
        self.ETA = 0.4
        self.PHI = 0.2
        self.DELTA = 0.85

        self.info_content()
    

    def info_content(self):
        self.N = 0
        self.brown_freqs = {}

        for sent in brown.sents:
            for word in sent:
                word = word.lower()
                if word not in self.brown_freqs:
                    self.brown_freqs[word] = 1 # TODO  Was 0
                self.brown_freqs[word] = self.brown_freqs[word] + 1
                self.N += 1
    
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


    def get_length_dist(self, synset_1, synset_2):

        if synset_1 is None or synset_2 is None:
            return 0.0
        if synset_1 == synset_2:
            return 1
        
        wset1 = set([str(x.name) for x in synset_1.lemmas()])
        wset2 = set([str(x.name) for x in synset_2.lemmas()])

        if len(wset1.intersection(wset2)) > 0:
            return np.exp(-1 * self.ALPHA)
        
        dist = synset_1.shortest_path_distance(synset_2)

        if dist is None:
            return 1

        return np.exp(-dist * self.ALPHA)
    
    def hyperbolic_ctg(self, val):
        return (math.exp(self.BETA * val) - math.exp(- self.BETA * val)) / (math.exp(self.BETA * val) + math.exp(- self.BETA * val))

    def hierarchy_dist(self, synset1, synset2):
        if synset1 is None or synset2 is None:
            return sys.maxsize()
        
        if synset1 == synset2:
            h_dist = max([x[1] for x in synset1.hypernym_distances()])
            return math.tanh(h_dist)
        
        pass


    def most_similar_words(self):
        pass

    

if __name__ == "__main__":
    main = Main()
    # print(f"BEST PAIR: {main.get_best_synset_pair('RAM', 'memory')}")

    # synsets_word1 = wn.synsets("RAM")
    # print(synsets_word1[0])
    # print( set([str(x.name()) for x in synsets_word1[0].lemmas()]))

    synsets_word1 = wn.synsets("RAM")
    fsynset = synsets_word1[0]
    ssynset = synsets_word1[1]

    print(f"{synset.shortest_path_distance(ssynset)}")
