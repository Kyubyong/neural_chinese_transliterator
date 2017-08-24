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
from hyperparams import Hyperparams as hp
import codecs
import pickle

def build_vocab():
    """Builds vocabulary from the corpus.
    Creates a pickle file and saves vocabulary files (dict) to it.
    """
    from collections import Counter
    from itertools import chain

    # pinyin
    if hp.isqwerty:
        pnyns = u"EUabcdefghijklmnopqrstuvwxyz0123456789。，！？" #E: Empty, U: Unknown
        pnyn2idx = {pnyn:idx for idx, pnyn in enumerate(pnyns)}
        idx2pnyn = {idx:pnyn for idx, pnyn in enumerate(pnyns)}
    else:
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
    
    if hp.isqwerty:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.qwerty.pkl', 'wb'))
    else:
        pickle.dump((pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi), open('data/vocab.nine.pkl', 'wb'))

if __name__ == '__main__':
    build_vocab(); print("Done" )
