# -*- coding: utf-8 -*-
# @Time    : 18-8-24 下午7:09
# @Author  : HeJi
# @FileName: hsi_bert.py
# @E-mail: hj@jimhe.cn

import tensorflow as tf
from module import label_smoothing, transformer_encoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from utils import get_gta_v4, convert_to_one_hot, shrink_labels
from sklearn.metrics import classification_report
import seaborn
import matplotlib.pyplot as plt
import optimization
from module import feedforward
import copy
from bert_module import embedding_postprocessor, create_initializer, layer_norm, create_attention_mask_from_input_mask
from dataset import simple_data_generator

def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax)

class HSI_BERT(object):

    def __init__(self, max_len, n_channel, max_depth = 1, num_head = 5, num_hidden = 200, drop_rate = 0.3, num_classes = 16, start_learning_rate = 1e-3, prembed = False, prembed_dim = 50, masking = False, pooling = False, pool_size = 3):
        self.max_len = max_len
        self.n_channel = n_channel
        self.max_depth = max_depth
        self.num_head = num_head
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.graph = tf.Graph()
        self.start_learning_rate = start_learning_rate
        self.sess = tf.Session(graph=self.graph)
        self.prembed = prembed
        self.prembed_dim = prembed_dim
        self.num_hidden = num_hidden
        self.masking = masking
        self.pooling = pooling
        self.pool_size = pool_size


    def build(self, is_training = True):
        print("#"*100)
        print(self.prembed)
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.max_len, self.n_channel])
            self.y = tf.placeholder(tf.int32, [None, ])

            x_shape = tf.shape(self.x)
            num_data = x_shape[0]
            masks = None
            if self.masking:
                print("*"*100)
                print("Masking")
                pitch_size = int(np.sqrt(self.max_len))
                single_mask = self.get_mask(pitch_size=pitch_size)
                single_mask = single_mask.flatten()
                single_mask = tf.convert_to_tensor(np.expand_dims(single_mask, axis=0), dtype=tf.bool)
                masks = tf.tile(single_mask, [num_data, 1])
                masks = create_attention_mask_from_input_mask(self.x, masks)
            print("self.prembed", self.prembed)
            emb_in = self.x
            #emb_in = embedding_postprocessor(emb_in, max_position_embeddings=200)
            if self.prembed:
                print("#"*100)
                emb_in = tf.layers.dense(emb_in, self.prembed_dim, activation=tf.tanh, use_bias=True)
                #emb_in = feedforward(emb_in, [self.num_hidden, self.prembed_dim])
                #emb_in = tf.layers.dropout(emb_in, rate=self.drop_rate)
                #emb_in = layer_norm(emb_in)
            print("my_emb_dim", emb_in.get_shape().as_list()[-1])
            #emb_dim = tf.shape(emb_in)[-1]
            emb_dim = emb_in.get_shape().as_list()[-1]
            print("emb_dim", emb_dim)
            #emb_in = tf.reshape(emb_in, [num_classes*num_support, emb_dim])
            attens = []
            for i in range(self.max_depth):
                name_scope = "Transformer_Encoder_"+str(i)
                if i == 0:
                    enc_embs, atten = transformer_encoder(emb_in, num_units=emb_dim,num_heads=self.num_head, num_hidden=self.num_hidden, dropout_rate=self.drop_rate, mask=masks, scope=name_scope)
                else:
                    enc_embs, atten = transformer_encoder(enc_embs, num_units=emb_dim, num_heads=self.num_head,num_hidden=self.num_hidden, dropout_rate=self.drop_rate,mask=masks, scope=name_scope)
                attens.append(atten)
            #emb_x = tf.reshape(emb_in, [1, num_classes*num_support, self.im_height*self.im_width*self.channels])
            if self.pooling:
                print("*"*100)
                print("Pooling")
                pooled_data = self.simple_pooler(enc_embs)#self.rect_pooler(enc_embs)
                enc_embs = tf.reshape(pooled_data, [num_data, emb_dim])
            else:
                print("8"*100)
                print("No Pooling")
                enc_embs = tf.reshape(enc_embs, [num_data, self.max_len*emb_dim])

            self.enc_embs = enc_embs
            self.istarget = tf.to_float(tf.not_equal(self.y, -99))

            self.logits = tf.layers.dense(enc_embs, self.num_classes)
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))) / tf.reduce_sum(self.istarget)
            self.attens = attens
            tf.summary.scalar('acc', self.acc)
            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.num_classes))
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)
                print("self.loss")
                self.mean_loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.istarget)

                # Training Scheme
                #self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #self.learning_rate = tf.train.exponential_decay(
                #    learning_rate=starter_learning_rate, global_step=self.global_step, decay_steps=20, decay_rate=0.95, staircase=False)
                #self.learning_rate = tf.constant(self.start_learning_rate)
                #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)#tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.98, epsilon=1e-8)
                #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                #self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                self.train_op, self.learning_rate = optimization.create_optimizer(
                    self.mean_loss, self.start_learning_rate, num_train_steps = 10000, num_warmup_steps=200, use_tpu = False)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                tf.summary.scalar('learning rate', self.learning_rate)
                self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver()

    def fit(self, X, y, X_vali = None, y_vali = None, batch_size = 24, nb_epochs = 10,
                      log_every_n_samples = 50, save_path = None, patience = 10):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        least_mean_loss = 10000
        bad_round = 0
        num_batches = len(y) // batch_size
        for epoch in range(nb_epochs):
            train_generator = simple_data_generator(X, y, batch_size=batch_size, shuffle=True)

            for ind,(X_batch, y_batch) in enumerate(train_generator):

                my_feed_dict = {self.x: X_batch,
                                self.y: y_batch.flatten(),
                                }
                _, preds, a, ls, ac, lr = self.sess.run([self.train_op, self.preds, self.attens, self.mean_loss, self.acc, self.learning_rate],
                                              feed_dict=my_feed_dict)
                if (ind + 1) % log_every_n_samples == 0:
                    #print(ls, ac)
                    print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epoch + 1, nb_epochs,
                                                                                                 ind + 1,num_batches,
                                                                                                 ls, ac))
                    #print(classification_report(y_true=support.y.flatten(), y_pred=preds))
                    print("learning rate",lr)
                    #print(a[0][0].shape)
                    #print(a[0][0][:,:,0])
                    #print(support.y.flatten())
            if (X_vali is not None) and (y_vali is not None):
                mean_loss = self.get_loss(X_vali, y_vali)
                if mean_loss < least_mean_loss:
                    bad_round = 0
                    print("mean loss improve from %f to %f" % (least_mean_loss, mean_loss))
                    least_mean_loss = mean_loss
                    if save_path is not None:
                        print("saving model")
                        self.save(save_path)
                else:
                    bad_round += 1
                    if bad_round > patience:
                        print("breaking training")
                        break
                    print("mean loss is %f doesn't improve from %f" % (mean_loss, least_mean_loss))

        if (X_vali is None) or (y_vali is None):
            if save_path:
                self.save(save_path)


        for layer in range(self.max_depth):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for h in range(4):
                draw(a[layer][0][:,:,h],
                     range(25), range(25) if h == 0 else [], ax=axs[h])
            plt.show()


    def fit_generator(self, generator, vali_generator = None, nb_epochs = 10,
                      log_every_n_samples = 50, save_path = None, patience = 10):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        least_mean_loss = 10000
        bad_round = 0
        num_batches = len(generator)
        for epoch in range(nb_epochs):

            for ind in range(num_batches):
                X_batch = generator[ind][0]
                y_batch = generator[ind][1]

                my_feed_dict = {self.x: X_batch,
                                self.y: y_batch.flatten(),
                                }
                _, preds, a, ls, ac, lr = self.sess.run([self.train_op, self.preds, self.attens, self.mean_loss, self.acc, self.learning_rate],
                                              feed_dict=my_feed_dict)
                if (ind + 1) % log_every_n_samples == 0:
                    #print(ls, ac)
                    print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epoch + 1, nb_epochs,
                                                                                                 ind + 1,num_batches,
                                                                                                 ls, ac))
                    #print(classification_report(y_true=support.y.flatten(), y_pred=preds))
                    print("learning rate",lr)
                    #print(a[0][0].shape)
                    #print(a[0][0][:,:,0])
                    #print(support.y.flatten())
            generator.on_epoch_end()
            if vali_generator is not None:
                mean_loss = self.get_loss_from_generator(vali_generator)
                if mean_loss < least_mean_loss:
                    bad_round = 0
                    print("mean loss improve from %f to %f" % (least_mean_loss, mean_loss))
                    least_mean_loss = mean_loss
                    if save_path is not None:
                        print("saving model")
                        self.save(save_path)
                else:
                    bad_round += 1
                    if bad_round > patience:
                        print("breaking training")
                        break
                    print("mean loss is %f doesn't improve from %f" % (mean_loss, least_mean_loss))

        if vali_generator is None:
            if save_path:
                self.save(save_path)


        for layer in range(self.max_depth):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for h in range(4):
                draw(a[layer][0][:,:,h],
                     range(25), range(25) if h == 0 else [], ax=axs[h])
            plt.show()

    def predict(self, X):
        y_preds = []
        batch_size = 1024
        num_batches = len(X) // batch_size
        for i in range(num_batches):

            preds = self.sess.run(self.preds, feed_dict={self.x: X[i*batch_size:(i+1)*batch_size]})

            y_preds.append(preds)
        preds = self.sess.run(self.preds, feed_dict={self.x:X[batch_size*num_batches:]})

        y_preds.append(preds)
        y_preds = np.concatenate(y_preds,axis=0)
        return y_preds

    def predict_from_generator(self, generator):
        y_preds = []
        num_batches = len(generator)
        for ind in range(num_batches):
            X_batch = generator[ind][0]
            preds = self.sess.run(self.preds, feed_dict={self.x: X_batch})
            y_preds.append(preds)
        y_preds= np.concatenate(y_preds, axis=0)
        return y_preds


    def get_loss(self, X, y):
        batch_size = 1024
        num_batches = len(X) // batch_size
        mean_loss = 0
        for i in range(num_batches):
            mean_loss += self.sess.run(self.mean_loss, feed_dict={self.x: X[i*batch_size:(i+1)*batch_size],
                                                                  self.y: y[i*batch_size:(i+1)*batch_size].flatten()})
        mean_loss += self.sess.run(self.mean_loss, feed_dict={self.x: X[batch_size*num_batches:],
                                                              self.y: y[batch_size*num_batches:].flatten()})
        if num_batches != 0:
            mean_loss = mean_loss / num_batches

        return mean_loss

    def get_loss_from_generator(self, generator):
        loss = 0
        num_batches = len(generator)
        for ind in range(num_batches):
            X_batch = generator[ind][0]
            y_batch = generator[ind][1]
            loss += self.sess.run(self.mean_loss, feed_dict={self.x: X_batch,
                                                        self.y: y_batch})
        if ind != 0:
            loss = loss / ind
        return loss

    def get_logits(self, X):
        batch_size = 1024
        logits = []
        num_batches = len(X) // batch_size
        for i in range(num_batches):
            logit = self.sess.run(self.logits, feed_dict={self.x: X[i * batch_size:(i + 1) * batch_size]})
                                                          #self.coords:dataset.w[i*batch_size:(i+1)*batch_size]})
            logits.append(logit)
        logit = self.sess.run(self.logits, feed_dict={self.x: X[batch_size * num_batches:]})
        logits.append(logit)
        logits = np.concatenate(logits,axis=0)
        return logits

    def get_logits_from_generator(self, generator):
        logits = []
        num_batches = len(generator)
        for ind in range(num_batches):
            X_batch = generator[ind][0]
            logit = self.sess.run(self.logits, feed_dict={self.x: X_batch})
            logits.append(logit)
        logits = np.concatenate(logits,axis=0)
        return logits

    def get_attention(self, X, layer = 0):
        all_attentions = []
        batch_size = 1024
        num_batches = len(X) // batch_size
        for i in range(num_batches):
            batch_attens = self.sess.run(self.attens[layer], feed_dict={self.x: X[i * batch_size:(i + 1) * batch_size]})
                                                                        #self.coords: dataset.w[i * batch_size:(i + 1) * batch_size]})
            all_attentions.append(batch_attens)
        batch_attens = self.sess.run(self.attens[layer], feed_dict={self.x: X[batch_size * num_batches:]})
                                                                    #self.coords:dataset.w[batch_size * num_batches:]})
        all_attentions.append(batch_attens)
        return np.concatenate(all_attentions, axis=0)

    def get_attention_from_generator(self, generator, layer = 0):
        all_attentions = []
        num_batches = len(generator)
        for ind in range(num_batches):
            X_batch = generator[ind][0]
            batch_attens = self.sess.run(self.attens[layer],
                                         feed_dict={self.x: X_batch})
            all_attentions.append(batch_attens)
        all_attentions = np.concatenate(all_attentions, axis=0)
        return all_attentions


    def save(self, path):
        """
        save current session to the path
        :param sess: tf.Session()
        :param path: the path you want to save the model
        :return: None
        """
        self.saver.save(sess=self.sess, save_path=path)


    def restore(self, path):
        """
        :param sess: tf.Session()
        :param path: path containing stored sessions
        :return: None
        """
        self.build()
        new_path = os.path.split(path)[0]
        meta_data_path = path + '.meta'
        # print("meta_data_path",meta_data_path)
        saver = tf.train.import_meta_graph(meta_data_path)
        # print("new_path", new_path)
        saver.restore(self.sess, tf.train.latest_checkpoint(new_path))

    def simple_pooler(self, data):
        margin = int((self.max_len - 1) / 2)
        return tf.squeeze(data[:, margin:margin+1, :], axis=1)

    def rect_pooler(self, data):
        data_shape = tf.shape(data)
        num_data = data_shape[0]
        shape = data.get_shape().as_list()
        emb_dim = shape[-1]
        pitch_len = int(np.sqrt(shape[1]))
        mask = self.get_mask(pitch_size=pitch_len)
        new_data = tf.reshape(data, [num_data, pitch_len, pitch_len, emb_dim])
        return tf.boolean_mask(new_data, mask=mask, axis=1), mask

    def get_mask(self, pitch_size = 9):
        mask = np.zeros([pitch_size, pitch_size], dtype=np.bool)
        margin_g = int((pitch_size - 1) / 2)
        tcoord = [margin_g, margin_g]
        margin = int((self.pool_size - 1) / 2)
        mask[tcoord[0] - margin: tcoord[0] + margin + 1, tcoord[1] - margin
                                                         :tcoord[1] + margin + 1] = True
        return mask
