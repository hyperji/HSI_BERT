# -*- coding: utf-8 -*-
# @Time    : 18-8-24 上午12:22
# @Author  : HeJi
# @FileName: utils.py
# @E-mail: hj@jimhe.cn

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
import gc
from contextlib import contextmanager
import time

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


def show_image_by_label(all_imgs, all_labels, label, n_show_sample = 5):
    mask = all_labels == label
    associated_imgs = all_imgs[mask]
    random_index = np.random.choice(len(associated_imgs), (n_show_sample, ), replace=False)
    associated_imgs = associated_imgs[random_index]
    #fig = plt.figure
    for img in associated_imgs:
        plt.figure()
        plt.imshow(image.array_to_img(img).resize((200,200)))

def get_gta(one_hot_test_label, ont_hot_support_label, scale = 50):
    y_test = tf.reshape(tf.argmax(one_hot_test_label, axis=1),[-1])
    y_support = tf.reshape(tf.argmax(ont_hot_support_label, axis=1),[-1])
    gtas = []
    for i in range(y_test.get_shape()[0]):
        gtas.append(tf.to_float(tf.equal(y_test[i], y_support)))
    gta = tf.stack(gtas)
    return gta / scale

def get_gta_v2(one_hot_query_label, ont_hot_support_label, scale = 50):
    y_query = tf.reshape(tf.argmax(one_hot_query_label, axis=1), [-1])
    y_support = tf.reshape(tf.argmax(ont_hot_support_label, axis=1), [-1])
    gtas = []
    y_query_unstacked = tf.unstack(y_query)
    for single_y_query in y_query_unstacked:
        gtas.append(tf.to_float(tf.equal(single_y_query, y_support)))
    gta = tf.stack(gtas)
    return gta / scale

def get_gta_v3(one_hot_query_label, ont_hot_support_label, scale = 50):
    y_query = tf.cast(tf.reshape(tf.argmax(one_hot_query_label, axis=1), [-1]), tf.float32)
    y_support = tf.cast(tf.reshape(tf.argmax(ont_hot_support_label, axis=1), [-1]), tf.float32)
    gtas = tf.map_fn(lambda x:tf.to_float(tf.equal(x, y_support)), y_query)
    return tf.stack(gtas) / scale

def get_gta_v4(y_query, y_support, scale = 50):
    y_query = tf.cast(y_query, tf.float32)
    y_support = tf.cast(y_support, tf.float32)
    gtas = tf.map_fn(lambda x: tf.to_float(tf.equal(x, y_support)), y_query)
    return tf.stack(gtas) / scale


def shrink_labels(labels, num_head):
    labels = np.expand_dims(labels, axis=-1)
    labels = np.concatenate(np.split(labels, num_head, axis=0), axis=-1)
    labels = np.concatenate([labels[0,:,:]],axis=0)
    labels = np.squeeze(labels)
    return labels.T

def convert_to_one_hot(y, C):
    y = y.astype(int)
    return np.eye(C)[y.reshape(-1)]


def get_train_test(data, data_labels, used_labels = None, test_size = None, limited_num = None, shuffle = True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    unique_labels = np.unique(data_labels)
    if used_labels is not None:
        unique_labels = used_labels
    for index, label in enumerate(unique_labels):
        masks = (data_labels == label)
        length = masks.sum()
        if test_size:
            assert (limited_num is None)
            nb_test = int(test_size * length)
            test_indexes = np.random.choice(length, (nb_test,), replace=False)
            train_indexes = np.array([i for i in range(length) if i not in test_indexes])
        if limited_num:
            assert (test_size is None)
            nb_train = limited_num
            train_indexes = np.random.choice(length, (nb_train,), replace=False)
            test_indexes = np.array([i for i in range(length) if i not in train_indexes])
        if used_labels is not None:
            train_labels.extend([index]*len(train_indexes))
            test_labels.extend([index]*len(test_indexes))
        else:
            train_labels.extend(data_labels[masks][train_indexes])
            test_labels.extend(data_labels[masks][test_indexes])
        train_data.append(data[masks][train_indexes])
        test_data.append(data[masks][test_indexes])
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.array(test_labels)
    if shuffle:
        train_shuffle = np.random.permutation(len(train_labels))
        train_data = train_data[train_shuffle]
        train_labels = train_labels[train_shuffle]
        test_shuffle = np.random.permutation(len(test_labels))
        test_data = test_data[test_shuffle]
        test_labels = test_labels[test_shuffle]
    return train_data, train_labels, test_data, test_labels


def get_coordinates_labels(y_hsi):
    max_label = np.max(y_hsi)
    row_coords = []
    col_coords = []
    labels = []
    for lbl in range(1, max_label+1):
        real_label = lbl - 1
        lbl_locs = np.where(y_hsi == lbl)
        row_coords.append(lbl_locs[0])
        col_coords.append(lbl_locs[1])
        length = len(lbl_locs[0])
        labels.append(np.array([real_label]*length))
    row_coords = np.expand_dims(np.concatenate(row_coords), axis=-1)
    col_coords = np.expand_dims(np.concatenate(col_coords), axis=-1)
    return np.concatenate([row_coords, col_coords], axis=-1), np.concatenate(labels)


from operator import truediv

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def euclidean_distance(a, b):
    """

    :param a: np.array
    :param b: np.array
    :return: a scale
    """
    return np.sqrt(np.sum((a-b)**2))