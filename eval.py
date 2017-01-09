#-*- coding: utf-8 -*-
#!/usr/bin/python2
'''
Test.
Given test sentences in `data/input.csv`,
it writes the results to `data/output_***.txt`, where *** equals a model name.
'''
from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import Hyperparams, load_vocab, load_test_data
from train import ModelGraph
import codecs
import pickle
import distance

def main():  
    g = ModelGraph(mode="test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        if Hyperparams.isqwerty: 
            save_path = "qwerty/asset/train/ckpt"
        else:    
            save_path = "nine/asset/train/ckpt"
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        mname = open(save_path + "/checkpoint", 'r').read().split('"')[1]
        
        nums, X, expected_list = load_test_data()
        pnyn2idx, idx2pnyn, hanzi2idx, idx2hanzi = load_vocab()
        
        with codecs.open('data/output_{}.txt'.format(mname), 'w', 'utf-8') as fout:
            cum_score = 0
            full_score = 0
            for step in range(len(X)//64 + 1):
                n = nums[step*64:(step+1)*64] #number batch
                x = X[step*64:(step+1)*64] # input batch
                e = expected_list[step*64:(step+1)*64] # batch of ground truth strings
                 
                # predict characters
                logits = sess.run(g.logits, {g.x: x})
                preds = np.squeeze(np.argmax(logits, -1))
 
                for nn, xx, pp, ee in zip(n, x, preds, e): # sentence-wise
                    got = ''
                    for xxx, ppp in zip(xx, pp): # character-wise
                        if xxx == 0: break
                        if xxx == 1 or ppp == 1:
                            got += "*"
                        else: 
                            got += idx2hanzi.get(ppp, "*")
                    got =  got.replace("_", "")  # Remove blanks 
                     
                    error = distance.levenshtein(ee, got)
                    score = len(ee) - error
                    cum_score += score
                    full_score += len(ee)
                      
                    fout.write(u"{}\t{}\t{}\t{}\n".format(nn, ee, got, score))
            fout.write(u"Total acc.: {}/{}={}\n".format(cum_score, 
                                                        full_score, 
                                                        round(float(cum_score)/full_score, 2)))
                                        
if __name__ == '__main__':
    main()
    print("Done")