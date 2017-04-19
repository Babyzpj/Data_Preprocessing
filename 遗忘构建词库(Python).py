import pymongo
from tqdm import tqdm

db = pymongo.MongoClient().weixin.text_articles

def texts():
   for t in db.find().limit(30000):
       yield t['text']



import numpy as np
from collections import defaultdict

class Forget:
    def __init__(self, alpha=-np.log(0.254)/(7*60*60*24*6)):
        self.alpha = alpha
        self.chars = defaultdict(lambda: (0, 0))
        self.couples = defaultdict(lambda: (0, 0))
        self.total = 0.
    def get_char(self, char):
        self.chars[char] = self.chars[char][0]*np.exp(-self.alpha*(self.total-self.chars[char][1])), self.total
        return self.chars[char][0]
    def get_couple(self, couple):
        self.couples[couple] = self.couples[couple][0]*np.exp(-self.alpha*(self.total-self.couples[couple][1])), self.total
        return self.couples[couple][0]
    def add_char_couple(self, char, couple):
        self.total += 1
        self.chars[char] = self.get_char(char)+1, self.total
        self.couples[couple] = self.get_couple(couple)+1, self.total
    def clean(self):
        self.chars = defaultdict(lambda: (0, 0), {i:j for i,j in self.chars.iteritems() if self.get_char(i) >= 1.})
        self.couples = defaultdict(lambda: (0, 0), {i:j for i,j in self.couples.iteritems() if len(i)==2 and self.get_couple(i) >= 1.})
    def build_strong_couples(self):
        self.clean()
        N = sum([i[0] for i in self.chars.itervalues()])**2/sum([i[0] for i in self.couples.itervalues()])
        self.strong_couples = set([i for i,j in self.couples.iteritems() if N*j[0]/(self.chars[i[0]][0]*self.chars[i[1]][0]) > 1.])
    def build_vocabs(self, texts):
        for t in texts:
            for i in range(len(t)):
                self.add_char_couple(t[i], t[i:i+2])
    def find_words(self, texts, min_count=5):
        self.build_strong_couples()
        self.words = defaultdict(int)
        for t in texts:
            if t:
                tmp = [t[0]]
                for c in t:
                    if tmp[-1][-1]+c in self.strong_couples:
                        tmp[-1] += c
                    else:
                        tmp.append(c)
                for w in tmp:
                    self.words[w] += 1
        self.words = {i:j for i,j in self.words.iteritems() if j >= min_count}
    

forget = Forget()
forget.build_vocabs(tqdm(texts()))
forget.find_words(tqdm(texts()))

import pandas as pd
w = pd.Series(forget.words).sort_values(ascending=False)

