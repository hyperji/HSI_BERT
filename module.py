# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午3:29
# @Author  : HeJi
# @FileName: module.py
# @E-mail: hj@jimhe.cn

from __future__ import print_function
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import copy, math
import bert_module

def conv_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv


def encoder(x, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1')
        net = conv_block(net, h_dim, name='conv_2')
        #net = conv_block(net, h_dim, name='conv_3')
        #net = conv_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net


def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


def cosine_distances(test, support):
    """Computes pairwise cosine distances between provided tensors

    Parameters
    ----------
    test: tf.Tensor
      Of shape (n_test, n_feat)
    support: tf.Tensor
      Of shape (n_support, n_feat)

    Returns
    -------
    tf.Tensor:
      Of shape (n_test, n_support)
    """


    rnorm_test = tf.rsqrt(
        tf.reduce_sum(tf.square(test), 1, keepdims=True)) + 1e-7
    rnorm_support = tf.rsqrt(
        tf.reduce_sum(tf.square(support), 1, keepdims=True)) + 1e-7
    test_normalized = test * rnorm_test
    support_normalized = support * rnorm_support
    try:
        support_normalized_t = tf.transpose(support_normalized, perm=[1, 0])
    except:
        print("support_normalized", support_normalized)
        print("error")
    g = tf.matmul(test_normalized, support_normalized_t)  # Gram matrix
    return g


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = (key.get_shape().as_list()[-1] ** 0.5)
    scores = tf.matmul(query, tf.transpose(key, [0, 2, 1])) / d_k
    print("scores", scores)
    if mask is not None:
        adder = (1.0 - tf.cast(mask, tf.float32)) * -10000.0
        scores += adder
        #scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = tf.nn.softmax(scores, axis=-1)
    print("p_atten", p_attn)
    if dropout is not None:
        p_attn = tf.layers.dropout(p_attn, rate=dropout)
    return tf.matmul(p_attn, value), p_attn


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=4,
                        dropout = 0.1,
                        mask = None,
                        scope="multihead_attention",
                        reuse=False):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    # with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            print("the num units is None")
            num_units = queries.get_shape().as_list()[-1]
        print("query", queries)
        print('num_heads', num_heads)
        print("num_units", num_units)
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.tanh, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.tanh, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.tanh, use_bias=False)  # (N, T_k, C)
        print("Q.get_shape().as_list()", Q.get_shape().as_list())
        print("K.get_shape().as_list()", K.get_shape().as_list())
        print("V.get_shape().as_list()", V.get_shape().as_list())
        print("Q", Q)

        # Split and concat
        # Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        # K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        # V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)

        print("Q_.get_shape().as_list()", Q_.get_shape().as_list())
        print("K_.get_shape().as_list()", K_.get_shape().as_list())
        print("V_.get_shape().as_list()", V_.get_shape().as_list())

        if mask is not None:
            mask = tf.tile(mask, [num_heads, 1, 1])

        outputs, atten = attention(Q_, K_, V_, mask=mask, dropout=dropout)
        print("omhomg", atten)
        atten = tf.split(atten, num_heads, axis=0)
        atten = [tf.expand_dims(a,axis=-1) for a in atten]
        atten = tf.concat(atten,axis=-1)


        print("atten", atten)

        '''
        if masking:
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        '''
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)  # (N, T_q, C)

        outputs = tf.layers.dense(outputs, num_units, activation=tf.tanh, use_bias=False)

        return outputs, atten


def transformer_encoder(emb_in,
                        num_units=None,
                        num_heads=4,
                        num_hidden = 200,
                        dropout_rate=0.3,
                        attention_dropout = 0.1,
                        mask = None,
                        is_training=True,
                        scope="Transformer_encoder",
                        reuse=False):
    emb_dim = emb_in.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):

        embs, atten = multihead_attention(emb_in, emb_in, num_units=num_units,
                                   num_heads=num_heads,dropout=attention_dropout, mask=mask)
        embs += emb_in

        embs = tf.layers.dropout(embs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        embs = bert_module.layer_norm(embs)

        transition_embs = feedforward(embs, num_units=[num_hidden, emb_dim])
        embs = embs + transition_embs

        embs = tf.layers.dropout(embs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        embs = bert_module.layer_norm(embs)

        return embs, atten


def transformer_decoder(queries,
                        keys,
                        num_units=None,
                        num_heads=4,
                        dropout_rate=0.3,
                        mask = None,
                        is_training = True,
                        scope="transformer_decoder",
                        reuse=False):
    emb_dim = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        query_embs = multihead_attention(queries, keys, num_units=num_units,
                                         num_heads=num_heads, mask=mask)
        query_embs += queries
        query_embs = tf.layers.dropout(query_embs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        query_embs = bert_module.layer_norm(query_embs)

        transition_embs = feedforward(query_embs, num_units=[4 * emb_dim, emb_dim])
        query_embs = query_embs + transition_embs

        query_embs = tf.layers.dropout(query_embs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        query_embs = bert_module.layer_norm(query_embs)

        return query_embs


def positional_encoding_v2(inputs,
                           max_len,
                           num_units,
                           scope = "positional_encoding",
                           reuse = None):

    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        pe = np.zeros([max_len, num_units])

        position = np.expand_dims(np.arange(0, max_len),axis=1)
        div_term = np.exp(np.arange(0, num_units, 2) *
                             -(math.log(10000.0) / num_units))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, axis=0)
        return tf.cast(pe[:, :input_shape[1]], tf.float32)


def positional_encoding(inputs,
                        num_units,
                        max_len,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.
    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    with tf.variable_scope(scope, reuse=reuse):
        #position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        position_ind = inputs

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(max_len)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs



def feedforward(inputs,
                num_units=[2048, 512],
                scope="transiting_function",
                reuse=tf.AUTO_REUSE):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    print("num_units", num_units)
    with tf.variable_scope(scope, reuse=reuse):

        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
              "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
              "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        """
        outputs = tf.layers.dense(inputs,num_units[0],activation=bert_module.gelu, use_bias=True)
        outputs = tf.layers.dense(outputs, num_units[1], use_bias=True)
        """
    return outputs

class KNN(object):
    """
    tensorflow version knn
    """

    def __init__(self, K = 1, metric = 'cosine', return_op = False):
        self.K = K
        self.metric = metric
        self.return_op = return_op
        if not self.return_op:
            self.sess = tf.Session()

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        if self.metric == 'cosine':
            g = cosine_distances(X, self.X)
        ng, indexs = tf.nn.top_k(g, self.K)
        preds = tf.gather(self.y, indexs)
        if not self.return_op:
            preds = self.sess.run(preds)
        return preds


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)
