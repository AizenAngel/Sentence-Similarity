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

# TODO: Razumeti implementaciju hierarchy_dist i potencijalno popraviti 

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

        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in self.brown_freqs:
                    self.brown_freqs[word] = 1 # TODO  Was 0
                self.brown_freqs[word] = self.brown_freqs[word] + 1
                self.N += 1


    def get_info_content_about_word(self, word):
        word = word.lower()
        return 0 if word not in self.brown_freqs else self.brown_freqs[word] 
    

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
                #print(f"{synset1}, {synset2} {similarity}")
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
    

    def get_hierarchy_dist(self, synset1, synset2):
        if synset1 is None or synset2 is None:
            return sys.maxsize()
        
        if synset1 == synset2:
            h_dist = max([x[1] for x in synset1.hypernym_distances()])
            return math.tanh(h_dist * self.BETA)
        
        hypernums1  = {x[0]: x[1] for x in synset1.hypernym_distances()}
        hypernums2  = {x[0]: x[1] for x in synset2.hypernym_distances()}
        lcs_candidates = set(hypernums1.keys()).intersection(set(hypernums2.keys()))

        if len(lcs_candidates) == 0:
            return math.tanh(0)
        
        lcs_dists = []
        for lcs_candidate in lcs_candidates:
            lcs_dists.append(max([hypernums1[lcs_candidate], hypernums2[lcs_candidate]]))
        
        return math.tanh(max(lcs_dists) * self.BETA)


    def get_word_similarity(self, word1, word2):
        synset_pair = self.get_best_synset_pair(word1, word2)
        hierarchy_dist_est = self.get_hierarchy_dist(synset_pair[0], synset_pair[1])
        length_dist_est = self.get_length_dist(synset_pair[0], synset_pair[1])

        return hierarchy_dist_est * length_dist_est



    def get_most_similar_word(self, word, word_set):
        max_sym = -1
        sym_word = None
        for ref_word in word_set:
            sym = self.get_word_similarity(word, ref_word)
            if sym > max_sym:
                sym_word = ref_word
                max_sym = sym

        return sym_word, max_sym


    def get_semantic_vector(self, words, joined_words_set):
        
        semantic_vector = np.zeros(len(joined_words_set))
        
        for (id, joined_word) in enumerate(joined_words_set):
            if joined_word in words:
                semantic_vector[id] = self.get_info_content_about_word(joined_word) ** 2
            else:
                _, max_sym = self.get_most_similar_word(joined_word, words)
                max_sym = 0 if max_sym < self.PHI else max_sym
                semantic_vector[id] = max_sym ** 2
        
        return semantic_vector


    def semantic_similarity(self, sentence1, sentence2):
        words1_set = set(nltk.tokenize(sentence1))
        words2_set = set(nltk.tokenize(sentence2))
        joined_words_set = words1_set.union(words2_set)

        sem_vec1 = self.get_semantic_vector(words1_set, joined_words_set)
        sem_vec2 = self.get_semantic_vector(words2_set, joined_words_set)

        return np.dot(sem_vec1, sem_vec2.T) / (np.linalg.norm(sem_vec1) * np.linalg.norm(sem_vec2))



if __name__ == "__main__":
    # main = Main()
    # print(f"BEST PAIR: {main.get_best_synset_pair('RAM', 'memory')}")

    # synsets_word1 = wn.synsets("RAM")
    # print(synsets_word1[0])
    # print( set([str(x.name()) for x in synsets_word1[0].lemmas()]))

    synsets_word1 = wn.synsets("RAM")[0]
    # fsynset1 = synsets_word1[0]
    # print(fsynset1.hypernym_distances())
    # print( set([str(x.name()) for x in fsynset1.lemmas()]))
    print(synsets_word1.root_hypernyms())

    print()

    synsets_word2 = wn.synsets("memory")[0]
    # fsynset2 = synsets_word2[0]
    # print(fsynset2.hypernym_distances())
    print(synsets_word2.root_hypernyms())

    synsets_word_100 = wn.synsets("number")[0]
    print(synsets_word_100.root_hypernyms())
