
import torch
import numpy as np
import sys, random
import time, datetime
import os
from pmi import cal_PMI


##DATA Helper##

class DataHelper(object):
    def __init__(self, mode='train', vocab=None):
        

        self.mode = mode

        self.base = os.path.join('data')

        self.current_set = os.path.join(self.base, '%s-stemmed.txt' % (self.mode))
        self.labels_str = 1
      
        content, label = self.get_content()

        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]
        self.label =  self.label_to_array(label)
        
        
    def label_to_array(self, label):
        num = []
        for l in label:
          if l != '':
            num.append(int(l))

        return np.array(num)



    def get_content(self):
        with open(self.current_set) as f:
            all = f.read()
            content = [line.split('\t') for line in all.split('\n')]

            cleaned_l = []
            cleaned_c = []

            for pair in (content):
                l_abel = pair[0][0:2]
                
                c_ontent = pair[0][3:]
                if c_ontent == '' or l_abel == '':
                    pass
                
                cleaned_c.append(c_ontent)
                cleaned_l.append(l_abel)

        label, content = zip([cleaned_l, cleaned_c])
        return content[0], label[0]
        

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['UNK']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab_5.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def count_word_freq(self, content):
        freq = dict(zip(self.vocab, [0 for i in range(len(self.vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                yield content, torch.tensor(label).cuda(), i
