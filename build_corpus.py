#-*- coding: utf-8 -*-
#!/usr/bin/python2
"""
Before running this code, make sure that you've downloaded Leipzig Chinese Corpus 
(http://corpora2.informatik.uni-leipzig.de/downloads/zho_news_2007-2009_1M-text.tar.gz)
Extract and copy the `zho_news_2007-2009_1M-sentences.txt` to `data/` folder.

This code should generate a file which looks like this:
2[Tab]zhegeyemianxianzaiyijingzuofei...。[Tab]这__个_页_面___现___在__已_经___作__废__...。

In each line, the id, pinyin, and a chinese sentence are separated by a tab.
Note that _ means blanks.

Created in Jan. 2017, kyubyong. kbpark.linguist@gmail.com
"""
from __future__ import print_function
import codecs
import os
import regex # pip install regex
from xpinyin import Pinyin # pip install xpinyin 

def align(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A tuple of pinyin and chinese sentence.
    '''
    pinyin = Pinyin()
    pnyns = pinyin.get_pinyin(sent, " ").split()
    
    hanzis = []
    for char, p in zip(sent.replace(" ", ""), pnyns):
        hanzis.extend([char] + ["_"] * (len(p) - 1))
        
    pnyns = "".join(pnyns)
    hanzis = "".join(hanzis)
    
    assert len(pnyns) == len(hanzis), "The hanzis and the pinyins must be the same in length."
    return pnyns, hanzis

def clean(text):
    if regex.search("[A-Za-z0-9]", text) is not None: # For simplicy, roman alphanumeric characters are removed.
        return ""
    text = regex.sub(u"[^ \p{Han}。，！？]", "", text)
    return text
    
def build_corpus():
    with codecs.open("data/zh.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/zho_news_2007-2009_1M-sentences.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: break
                
                try:
                    idx, sent = line.strip().split("\t")
                    sent = clean(sent)
                    if len(sent) > 0:
                        pnyns, hanzis = align(sent)
                        fout.write(u"{}\t{}\t{}\n".format(idx, pnyns, hanzis))
                except:
                    continue # it's okay as we have a pretty big corpus!
                
                if i % 10000 == 0: print(i, )
                i += 1

if __name__ == "__main__":
    build_corpus()
    print("Done")