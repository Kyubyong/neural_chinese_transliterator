#-*- coding: utf-8 -*-
#!/usr/bin/python2
'''
Preprocessing.
Note:
Nine key pinyin keyboard layout sample:

`      ABC   DEF
GHI    JKL   MNO
POQRS  TUV   WXYZ

'''
from __future__ import print_function
import codecs
import regex
import pickle
import numpy as np
import sys

class Hyperparams:
    '''Hyper parameters'''
    batch_size = 64
    embed_dim = 300
    maxlen = 50
    isqwerty = False # If False, 10 keyboard layout is assumed. 

def build_vocab():
    """Builds vocabulary from the corpus.
    Creates a pickle file and saves vocabulary files (dict) to it.
    """
    from collections import Counter
    from itertools import chain
    import os
    
    # pinyin
    if Hyperparams.isqwerty:
        if not os.path.exists("data/qwerty"): os.mkdir("data/qwerty")
        
        pnyns = u"EUabcdefghijklmnopqrstuvwxyz0123456789。，！？" #E: Empty, U: Unknown
        pnyn2idx = {pnyn:idx for idx, pnyn in enumerate(pnyns)}
        idx2pnyn = {idx:pnyn for idx, pnyn in enumerate(pnyns)}
    else:
        if not os.path.exists("data/nine"): os.mkdir("data/nine")
        
        pnyn2idx, idx2pnyn = dict(), dict()
        pnyns_list = ["E", "U", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz", 
                      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", u"。", u"，", u"！", u"？"] #E: Empty, U: Unknown
        for i, pnyns in enumerate(pnyns_list):
            for pnyn in pnyns:
                pnyn2idx[pnyn] = i
    
    # hanzis
    hanzi_sents = [line.split('\t')[2] for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    hanzi2cnt = Counter(chain.from_iterable(hanzi_sents))
    hanzis = [hanzi for hanzi, cnt in hanzi2cnt.items() if cnt > 5] # remove long-tail characters
    
    hanzis.remove("_")
    hanzis = ["E", "U", "_" ] + hanzis # 0: empty, 1: unknown, 2: blank
    hanzi2idx = {hanzi:idx for idx, hanzi in enumerate(hanzis)}
    idx2hanzi = {idx:hanzi for idx, hanzi in enumerate(hanzis)}
    
    if Hyperparams.isqwerty:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/qwerty/vocab.pkl', 'wb'))
    else:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/nine/vocab.pkl', 'wb'))

def load_vocab():
    if Hyperparams.isqwerty:
        return pickle.load(open('data/qwerty/vocab.pkl', 'rb'))
    else:
        return pickle.load(open('data/nine/vocab.pkl', 'rb'))
            
def create_train_data():
    '''Embeds and vectorize words in corpus'''
    pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
    
    print("pnyn vocabulary size is", len(pnyn2idx))
    print("hanzi vocabulary size is", len(hanzi2idx))
    
    print("# Vectorize")
    pnyn_sents = [line.split('\t')[1] for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    hanzi_sents = [line.split('\t')[2] for line in codecs.open('data/zh.tsv', 'r', 'utf-8').read().splitlines()]
    
    xs, ys = [], [] # vectorized sentences
    for pnyn_sent, hanzi_sent in zip(pnyn_sents, hanzi_sents):
        if 10 < len(pnyn_sent) <= Hyperparams.maxlen:
            x, y = [], []
            for pnyn in pnyn_sent:
                if pnyn in pnyn2idx:
                    x.append(pnyn2idx[pnyn])
                else: # <UNK>
                    x.append(1) 
            
            for hanzi in hanzi_sent:
                if hanzi in hanzi2idx:
                    y.append(hanzi2idx[hanzi])
                else: # <UNK>
                    y.append(1)
            
            x.extend([0] * (Hyperparams.maxlen - len(x))) # zero post-padding
            y.extend([0] * (Hyperparams.maxlen - len(y))) # zero post-padding
            
            xs.append(x) 
            ys.append(y) 
 
    print("# Convert to 2d-arrays"    )
    X = np.array(xs)
    Y = np.array(ys)
    
    print("X.shape =", X.shape) # (104778, 50)
    print("Y.shape =", Y.shape) # (104778, 50)

    if Hyperparams.isqwerty:    
        np.savez('data/qwerty/X_Y.npz', X=X, Y=Y)
    else:
        np.savez('data/nine/X_Y.npz', X=X, Y=Y)

def load_train_data():
    '''Loads vectorized input training data
    '''
    if Hyperparams.isqwerty:    
        X, Y = np.load('data/qwerty/X_Y.npz')['X'], np.load('data/qwerty/X_Y.npz')['Y']
    else:
        X, Y = np.load('data/nine/X_Y.npz')['X'], np.load('data/nine/X_Y.npz')['Y']
    return X, Y

def load_test_data():
    '''Embeds and vectorize words in input corpus'''
    try:
        lines = [line for line in codecs.open('data/input.csv', 'r', 'utf-8').read().splitlines()[1:]]
    except IOError:
        raise IOError("Write the sentences you want to test line by line in `data/input.csv` file.")
     
    pnyn2idx, _, hanzi2idx, _ = load_vocab()
    
    nums = [] 
    xs = []
    expected_list = []
    for line in lines:
        num, pnyn_sent, expected = line.split(",")
        
        nums.append(num)
        expected_list.append(expected)
        
        x = []
        for pnyn in pnyn_sent[:Hyperparams.maxlen]:
            if pnyn in pnyn2idx: 
                x.append(pnyn2idx[pnyn])
            else:
                x.append(1) #"OOV", i.e., not converted.
         
        x.extend([0] * (Hyperparams.maxlen - len(x))) # zero post-padding
        xs.append(x)
     
    X = np.array(xs)
    return nums, X, expected_list

if __name__ == '__main__':
    build_vocab()
    create_train_data()
    print("Done" )