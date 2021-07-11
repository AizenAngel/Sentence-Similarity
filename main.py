from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import nltk
import sys
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
from termcolor import cprint


class Main:
    def __init__(self):
        self.ALPHA = 0.2
        self.BETA = 0.45
        self.ETA = 0.7
        self.PHI = 0.2
        self.DELTA = 0.85
        self.EPS = 1e-1

        self.info_content()


    def info_content(self):
        print("Calculating word frequencies for Brown Dataset...")
        self.N = 0
        self.brown_freqs = {}

        for sent in tqdm(brown.sents()):
            for word in sent:
                word = word.lower()
                if word not in self.brown_freqs:
                    self.brown_freqs[word] = 1 # TODO  Was 0
                self.brown_freqs[word] = self.brown_freqs[word] + 1
                self.N += 1


    def get_info_content_about_word(self, word):
        word = word.lower()
        return 0 if word not in self.brown_freqs else self.brown_freqs[word] 
    

    def get_color_for_similarity(self, similarity_measure):
        return "green" if similarity_measure > 0.75 else "yellow" if similarity_measure > 0.4 else  "red"


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
            return sys.maxsize
        
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



    def get_most_similar_word(self, word, word_corpus):
        max_sym = -1
        sym_word = None
        for ref_word in word_corpus:
            sym = self.get_word_similarity(word, ref_word)
            if sym > max_sym:
                sym_word = ref_word
                max_sym = sym

        return sym_word, max_sym


    def get_semantic_vector(self, words, joined_words):
        
        semantic_vector = np.zeros(len(joined_words))
        
        for (id, joined_word) in enumerate(joined_words):
            if joined_word in words:
                semantic_vector[id] = self.get_info_content_about_word(joined_word) ** 2
            else:
                _, max_sym = self.get_most_similar_word(joined_word, words)
                max_sym = 0 if max_sym < self.PHI else max_sym
                semantic_vector[id] = max_sym ** 2
        
        return semantic_vector


    def get_semantic_similarity(self, sentence1, sentence2):
        words1 = nltk.word_tokenize(sentence1)
        words2 = nltk.word_tokenize(sentence2)
        joined_words = sorted(list( set(words1) | set(words2) ))

        sem_vec1 = self.get_semantic_vector(words1, joined_words)
        sem_vec2 = self.get_semantic_vector(words2, joined_words)

        return np.dot(sem_vec1, sem_vec2.T) / (LA.norm(sem_vec1) * LA.norm(sem_vec2))


    def get_word_order_vector(self, words, joined_words):
        word_order_vector = np.zeros(len(joined_words))

        for (id, joined_word) in enumerate(joined_words):
            if joined_word in words:
                word_order_vector[id] = words.index(joined_word) + 1
            else:
                most_sym_word, max_sym = self.get_most_similar_word(joined_word, words)
                word_order_vector[id] = (words.index(most_sym_word) + 1) if max_sym > self.ETA else 0
        
        
        # print (f"Word order vector: {word_order_vector}")
        # print (f"Words: {words}")
        # print()

        return word_order_vector


    def get_word_order_similarity(self, sentence1, sentence2):
        words1 = nltk.word_tokenize(sentence1)
        words2 = nltk.word_tokenize(sentence2)
        joined_words = sorted(list(set(words1).union(set(words2))))  
        
        # print (f"Joined words: {joined_words}")   
        # print ()

        word_order_vector1 = self.get_word_order_vector(words1, joined_words)
        word_order_vector2 = self.get_word_order_vector(words2, joined_words)

        return 1 - (LA.norm(word_order_vector1 - word_order_vector2) / LA.norm(word_order_vector1 + word_order_vector2))


    def find_similarity(self, sentence1, sentence2):
        word_order_similarity = self.DELTA * self.get_word_order_similarity(sentence1, sentence2)
        semantic_similarity = (1 - self.DELTA) * self.get_semantic_similarity(sentence1, sentence2)
        
        #print(f"Word order similarity: {word_order_similarity}, semantic_similarity: {semantic_similarity}")

        similarity_measure =  semantic_similarity + word_order_similarity
        similarity_measure = 0 if similarity_measure < self.EPS else similarity_measure 
        color_for_similarity = self.get_color_for_similarity(similarity_measure)
        
        cprint(f"Similarity: {similarity_measure}", color_for_similarity) 

