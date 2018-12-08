# -*- coding: utf-8 -*-
# @Time    : 18-10-12 下午9:59
# @Author  : HeJi
# @FileName: grammar.py
# @E-mail: hj@jimhe.cn
from utils import euclidean_distance
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import random
import scipy
import copy
from utils import get_coordinates_labels, timer, get_train_test
"""
def select_round_superpixel(target_point, radius = 2):
    start_point = tuple([target_point[0] - radius, target_point[1] - radius])
    end_point = tuple([target_point[0] + radius, target_point[1] + radius])
    correct_points = []
    for i in range(start_point[0], end_point[0]+1):
        for j in range(start_point[1], end_point[1]+1):
            eval_point = np.zeros(2)
            if i < target_point[0]:
                eval_point[0]= i+0.5
            elif i > target_point[0]:
                eval_point[0] = i-0.5
            elif i == target_point[0]:
                eval_point[0] = i
            if j < target_point[1]:
                eval_point[1]= j+0.5
            elif j > target_point[1]:
                eval_point[1] = j-0.5
            elif j == target_point[1]:
                eval_point[1] = j
            if euclidean_distance(target_point, eval_point)<radius:
                correct_points.append(tuple([i, j]))
    return correct_points
"""


def select_round(img, target_coords, r):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)

    zeroPaddedX = padWithZeros(img, margin=r)
    new_target_coords = target_coords + r
    shape = zeroPaddedX.shape
    height = shape[0]
    width = shape[1]
    sentences = []
    for ntc in new_target_coords:
        mask = np.zeros((height, width), dtype=np.bool)
        y, x = np.ogrid[-ntc[0]:height - ntc[0], -ntc[1]:width - ntc[1]]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)
        sentences.append(np.expand_dims(zeroPaddedX[mask], axis=0))
    sentences = np.concatenate(sentences, axis=0)
    return sentences

def select_round_no_pad(img, target_coords, r):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    sentences = []
    for ntc in target_coords:
        mask = np.zeros((height, width), dtype=np.bool)
        y, x = np.ogrid[-ntc[0]:height - ntc[0], -ntc[1]:width - ntc[1]]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)
        sentences.append(np.expand_dims(img[mask], axis=0))
    sentences = np.concatenate(sentences, axis=0)
    return sentences



#  pad zeros to dataset
def padWithZeros(X, margin=2):
    return np.pad(X, [(margin, margin), (margin, margin), (0,0)], mode="constant")


#  apply PCA preprocessing for data sets
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


#  over sample
def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]),
                                                   axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

#  standartize
def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    #scaler = preprocessing.StandardScaler().fit(newX)
    newX = preprocessing.scale(newX)
    #newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX


def transform_array(array, mode = 0, degree = None):
    if (mode == 0):
        flipped_patch = np.flipud(array)
    if (mode == 1):
        flipped_patch = np.fliplr(array)
    if (mode == 2):
        assert(degree is not None)
        flipped_patch = scipy.ndimage.interpolation.rotate(array,
                                                               degree, axes=(1, 0), reshape=False, output=None,
                                                               order=3, mode='constant', cval=0.0, prefilter=False)
    noise = np.random.normal(0.0, 0.05, size=(array.shape))
    flipped_patch += noise
    return flipped_patch

def fuse_samples(sampleA, sampleB, alpha, beta):
    """

    :param sampleA: pixelA
    :param sampleB: pixelB
    :param alpha: size as sampleA
    :param beta: size as sampleA
    :return: fused sample
    """
    return alpha*sampleA+(1-alpha)*sampleB + beta

def generating_bands_samples(hsi, train_coordinates, train_labels, num_band_group = 10):
    new_hsi = copy.deepcopy(hsi)
    assert(new_hsi.shape[2]%num_band_group == 0)
    band_group_size = int(new_hsi.shape[2]/num_band_group)
    imgs = np.split(new_hsi, axis=2, indices_or_sections=num_band_group)
    unique_labels = np.unique(train_labels)
    for img in imgs:
        for lbl in unique_labels:
            lbl_train_coordinates = train_coordinates[train_labels==lbl]
            lbl_train_data = img[lbl_train_coordinates[:, 0], lbl_train_coordinates[:, 1], :]
            #means = np.mean(lbl_train_data, axis=0)
            stds = np.std(lbl_train_data, axis=0)
            for coord in lbl_train_coordinates:
                beta = np.array([np.random.normal(loc=0, scale=0.05) for i in range(new_hsi.shape[2])])
                #alpha = np.random.random(size=band_group_size)
                alpha = np.random.random()
                sampleA = img[coord[0], coord[1], :]
                bcoord = lbl_train_coordinates[np.random.choice(lbl_train_coordinates.shape[0])]
                sampleB = img[bcoord[0],bcoord[1],:]
                img[coord[0], coord[1], :] = fuse_samples(sampleA, sampleB, alpha, beta)
    return imgs


def generating_pitches_samples(pitches, labels):
    mypitches = copy.deepcopy(pitches)
    unique_labels = np.unique(labels)
    generated_pitches = []
    generated_labels = []
    for lbl in unique_labels:
        lbl_pitches = mypitches[labels == lbl]
        for pitchA in lbl_pitches:
            pitchB = lbl_pitches[np.random.choice(lbl_pitches.shape[0])]
            beta = np.array([np.random.normal(loc=0, scale=0.05) for i in range(mypitches.shape[-1])])
            # print("beta.shape", beta.shape)
            fused_sample = fuse_samples(sampleA=pitchA, sampleB=pitchB, alpha=np.random.random(), beta=beta)
            fused_sample = np.expand_dims(fused_sample, axis=0)
            generated_pitches.append(fused_sample)
            generated_labels.append(lbl)

    generated_pitches = np.concatenate(generated_pitches, axis=0)
    generated_labels = np.array(generated_labels)
    return generated_pitches, generated_labels



def rotation_and_flip(pitches, labels, shuffle = True):
    assert (len(pitches.shape) == 4)
    print("Applying rotation and flip")
    augment_pitches = []
    augment_labels = []
    for i, pitch in enumerate(pitches):
        p9 = transform_array(pitch, mode=2, degree=90)
        p18 = transform_array(pitch, mode=2, degree=180)
        p27 = transform_array(pitch, mode=2, degree=270)
        p9f = transform_array(p9, mode=0)
        p18f = transform_array(p18, mode=0)
        p27f = transform_array(p27, mode=0)

        pitch = np.expand_dims(pitch, axis=0)
        p9 = np.expand_dims(p9, axis=0)
        p18 = np.expand_dims(p18, axis=0)
        p27 = np.expand_dims(p27, axis=0)
        p9f = np.expand_dims(p9f, axis=0)
        p18f = np.expand_dims(p18f, axis=0)
        p27f = np.expand_dims(p27f, axis=0)

        pitch_augmented = np.concatenate([pitch,p9,p18,p27,p9f,p18f,p27f],axis=0)
        pitch_label_augmented = np.array([labels[i]]*7)
        augment_pitches.append(pitch_augmented)
        augment_labels.append(pitch_label_augmented)

    augment_pitches = np.concatenate(augment_pitches, axis=0)
    augment_labels = np.concatenate(augment_labels)

    indexes = np.arange(augment_pitches.shape[0])
    indexes = np.random.permutation(indexes)
    if shuffle:
        augment_pitches = augment_pitches[indexes]
        augment_labels = augment_labels[indexes]

    return augment_pitches, augment_labels


def zmm_random_flip(data,label,seed=0):
    print("Appling ZMM random flip")
    num = data.shape[0]
    datas = []
    labels = []
    assert (len(data.shape) == 4)
    for i in range(num):
        datas.append(data[i])
        noise = np.random.normal(0.0, 0.05, size=(data[i].shape))
        datas.append(np.fliplr(data[i]) + noise)
        labels.append(label[i])
        labels.append(label[i])
    datas = np.asarray(datas, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    np.random.seed(seed)
    index = np.random.permutation(datas.shape[0])
    return datas[index], labels[index]


def select_rect(hsi, target_coords, pitch_size = 5):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    margin = int((pitch_size - 1) / 2)
    zeroPaddedX = padWithZeros(hsi, margin=margin)

    ncoords = target_coords + margin

    pitches = [zeroPaddedX[ncoords[i,0] - margin:ncoords[i,0] + margin + 1, ncoords[i,1]-margin
                                                   :ncoords[i,1] + margin + 1] for i in range(len(ncoords))]
    pitches = [np.expand_dims(pitch, axis=0) for pitch in pitches]

    return np.concatenate(pitches, axis=0) #, pitch_coords


def select_rect_no_pad(hsi, target_coords, pitch_size = 5):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    margin = int((pitch_size - 1) / 2)

    pitches = [hsi[target_coords[i,0] - margin:target_coords[i,0] + margin + 1, target_coords[i,1]-margin
                                                   :target_coords[i,1] + margin + 1] for i in range(len(target_coords))]
    pitches = [np.expand_dims(pitch, axis=0) for pitch in pitches]

    return np.concatenate(pitches, axis=0)


def select_line(img, target_coords, length = 5, mod = "h"):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    margin = int((length - 1) / 2)
    zeroPaddedX = padWithZeros(img, margin=margin)
    ncoords = target_coords + margin
    if mod == "h":
        lines = [zeroPaddedX[ncoords[i, 0], ncoords[i, 1] - margin:ncoords[i, 1] + margin + 1] for i in range(len(ncoords))]
    elif mod =="v":
        lines = [zeroPaddedX[ncoords[i, 0] - margin:ncoords[i,0]+margin+1, ncoords[i,1]] for i in range(len(ncoords))]
    lines = [np.expand_dims(line, axis=0) for line in lines]
    return np.concatenate(lines, axis=0)



def select_line_no_pad(img, target_coords, length = 5, mod = "h"):
    if len(target_coords.shape) == 1:
        target_coords = np.expand_dims(target_coords, axis=0)
    margin = int((length - 1) / 2)
    if mod == "h":
        lines = [img[target_coords[i, 0], target_coords[i, 1] - margin:target_coords[i, 1] + margin + 1] for i in range(len(target_coords))]
    elif mod =="v":
        lines = [img[target_coords[i, 0] - margin:target_coords[i,0]+margin+1, target_coords[i,1]] for i in range(len(target_coords))]
    lines = [np.expand_dims(line, axis=0) for line in lines]
    return np.concatenate(lines, axis=0)



def Grammar(img, tgt_coords, method = "rect 11"):
    try:
        region_type , param = method.split()
        param = int(param)
    except:
        region_type = method
    if region_type == "rect":
        assert (param is not None)
        data = select_rect_no_pad(img, tgt_coords,pitch_size=param)
        data = np.reshape(data, [data.shape[0], data.shape[1]*data.shape[2], data.shape[3]])
    elif region_type == 'round':
        assert (param is not None)
        data = select_round_no_pad(img, tgt_coords, r=param)
    elif region_type == "dot":
        data = select_rect_no_pad(img, tgt_coords, pitch_size = 1)
        data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2], data.shape[3]])
    elif region_type == 'hl':
        assert (param is not None)
        data = select_line_no_pad(img, tgt_coords, length=param, mod="h")
    elif region_type == 'vl':
        assert (param is not None)
        data = select_line_no_pad(img, tgt_coords, length=param, mod="v")
    elif region_type =='cross':
        assert (param is not None)
        datah = select_line_no_pad(img, tgt_coords, length=param, mod="h")
        datav = select_line_no_pad(img, tgt_coords, length=param, mod="v")
        data = np.concatenate([datah, datav], axis=1)
    return data


