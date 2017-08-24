# -*- coding: utf-8 -*-
# !/usr/bin/python2
'''
Training.
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
from data_load import get_batch, load_vocab
import tensorflow as tf
from modules import *
from tqdm import tqdm


class Graph():
    '''Builds a model graph'''

    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch()
            else:  # Evaluation
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen,))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen,))

            # Load vocabulary
            pnyn2idx, _, hanzi2idx, _ = load_vocab()

            # Character Embedding for x
            enc = embed(self.x, len(pnyn2idx), hp.embed_size, scope="emb_x")

            # Encoder pre-net
            prenet_out = prenet(enc,
                                num_units=[hp.embed_size, hp.embed_size // 2],
                                is_training=is_training)  # (N, T, E/2)

            # Encoder CBHG
            ## Conv1D bank
            enc = conv1d_banks(prenet_out,
                               K=hp.encoder_num_banks,
                               num_units=hp.embed_size // 2,
                               is_training=is_training)  # (N, T, K * E / 2)

            ## Max pooling
            enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)

            ## Conv1D projections
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=tf.nn.relu, scope="norm1")
            enc = conv1d(enc, hp.embed_size // 2, 5, scope="conv1d_2")  # (N, T, E/2)
            enc = normalize(enc, type=hp.norm_type, is_training=is_training,
                            activation_fn=None, scope="norm2")
            enc += prenet_out  # (N, T, E/2) # residual connections

            ## Highway Nets
            for i in range(hp.num_highwaynet_blocks):
                enc = highwaynet(enc, num_units=hp.embed_size // 2,
                                 scope='highwaynet_{}'.format(i))  # (N, T, E/2)

            ## Bidirectional GRU
            enc = gru(enc, hp.embed_size // 2, True, scope="gru1")  # (N, T, E)

            ## Readout
            self.outputs = tf.layers.dense(enc, len(hanzi2idx), use_bias=False)
            self.preds = tf.to_int32(tf.arg_max(self.outputs, dimension=-1))

            if is_training:
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.outputs)
                self.istarget = tf.to_float(tf.not_equal(self.y, tf.zeros_like(self.y)))  # masking
                self.hits = tf.to_float(tf.equal(self.preds, self.y)) * self.istarget
                self.acc = tf.reduce_sum(self.hits) / tf.reduce_sum(self.istarget)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / tf.reduce_sum(self.istarget)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                tf.summary.scalar('acc', self.acc)
                self.merged = tf.summary.merge_all()


def train():
    g = Graph(); print("Training Graph loaded")

    with g.graph.as_default():
        # Training
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop(): break
                for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                # Write checkpoint files at every epoch
                gs = sess.run(g.global_step)
                sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))


if __name__ == '__main__':
    train(); print("Done")
