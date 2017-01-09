#-*- coding: utf-8 -*-
#!/usr/bin/python2
'''
Training.
'''
from __future__ import print_function
from prepro import Hyperparams, load_vocab, load_train_data
import sugartensor as tf
import sys

def get_batch_data():
    '''Makes batch queues from the data.
    Returns:
      A Tuple of x (Tensor), y (Tensor).
      x and y have the shape [batch_size, maxlen].
    '''
    # Load data
    X, Y = load_train_data()
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X, tf.int64), 
                                                  tf.convert_to_tensor(Y, tf.int64)])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=Hyperparams.batch_size, 
                                  capacity=Hyperparams.batch_size*64,
                                  min_after_dequeue=Hyperparams.batch_size*32, 
                                  allow_smaller_final_batch=False) 
    num_batch = len(X) // Hyperparams.batch_size
    
    return x, y, num_batch # (64, 50) int64, (64, 50) int64, 1636

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self, mode="train"):
        '''
        Args:
          is_train: Boolean. If True, backprop is executed.
        '''
        if mode == "train":
            self.x, self.y, self.num_batch = get_batch_data() # (64, 50) int64, (64, 50) int64, 1636
        else: # test
            self.x = tf.placeholder(tf.int64, [None, Hyperparams.maxlen])
        
        # make embedding matrix for input characters
        pnyn2idx, _, hanzi2idx, _ = load_vocab()
        
        self.emb_x = tf.sg_emb(name='emb_x', voca_size=len(pnyn2idx), dim=Hyperparams.embed_dim)
        self.enc = self.x.sg_lookup(emb=self.emb_x)
        
        with tf.sg_context(size=5, act='relu', bn=True):
            for _ in range(20):
                dim = self.enc.get_shape().as_list()[-1]
                self.enc += self.enc.sg_conv1d(dim=dim) # (64, 50, 300) float32
        
        # final fully convolutional layer for softmax
        self.logits = self.enc.sg_conv1d(size=1, dim=len(hanzi2idx), act='linear', bn=False) # (64, 50, 5072) float32
        if mode == "train":
            self.ce = self.logits.sg_ce(target=self.y, mask=True) # (64, 50) float32
            self.istarget = tf.not_equal(self.y, tf.zeros_like(self.y)).sg_float() # (64, 50) float32
            self.reduced_loss = self.ce.sg_sum() / self.istarget.sg_sum() # () float32
            tf.sg_summary_loss(self.reduced_loss, "reduced_loss")
            
def train():
    g = ModelGraph(); print("Graph loaded!")
    
    if Hyperparams.isqwerty: 
        save_dir = 'qwerty/asset/train'
    else:
        save_dir = 'nine/asset/train'
        
    tf.sg_train(lr=0.0001, lr_reset=True, log_interval=10, loss=g.reduced_loss, max_ep=50, 
                save_dir=save_dir, early_stop=False, max_keep=5, ep_size=g.num_batch)
     
if __name__ == '__main__':
    train(); print("Done")