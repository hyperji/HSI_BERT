# -*- coding: utf-8 -*-
# @Time    : 18-8-23 下午4:52
# @Author  : HeJi
# @FileName: wrapup.py
# @E-mail: hj@jimhe.cn

from hsi_bert import HSI_BERT
import numpy as np
import gc
import argparse
import scipy.io as scio
from utils import get_train_test, get_coordinates_labels, AA_andEachClassAccuracy
from module import KNN
from dataset import Data_Generator, zeropad_to_max_len
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
import tensorflow as tf
import os
from grammar import Grammar, standartizeData, rotation_and_flip, zmm_random_flip, padWithZeros
from utils import timer

selection_rules = ["rect 11"]# "round 4", "round 5", "round 6"]


def get_matrics(y_true, y_pred):
    oa = accuracy_score(y_pred=y_pred, y_true=y_true)
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    apc, aa = AA_andEachClassAccuracy(cm)
    kappa = cohen_kappa_score(y1=y_true, y2=y_pred)
    apc = np.expand_dims(apc, axis=0)
    result = {"oa":oa, 'aa':aa, "k":kappa, "apc":apc}
    return result



def get_args():
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--n_epochs', type=int, default=30, help='n_epochs')
    parser.add_argument("--max_depth", type=int, default=2, help="max_depth")
    parser.add_argument('--batch_size', type=int, default=128, help="num_batches")
    parser.add_argument("--num_head", type=int, default=10)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--log_every_n_samples", type=int, default=5)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--test_size",type=float)
    parser.add_argument("--start_learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset", type=str, default="IN")
    parser.add_argument("--prembed", type=bool, default=True)
    parser.add_argument("--prembed_dim", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="data/IN")
    parser.add_argument("--repeat_term", type=int, default=10)
    parser.add_argument("--is_valid", type=bool, default=False)
    parser.add_argument("--limited_num", type=int)
    parser.add_argument("--num_hidden", type=int, default=200)
    parser.add_argument("--masking", type=bool, default=False)
    parser.add_argument("--pooling", type=bool, default=False)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--data_augment", type = bool, default=False)
    parser.add_argument("--max_len", type=int,default=121)
    parser.add_argument("--test_region", type=str, default="rect 11")

    # 如: python xx.py --foo hello  > hello
    args = parser.parse_args()
    return args

def main():
    print(tf.__version__)

    arg = get_args()

    print("arg.is_valid", arg.is_valid, "type(arg.is_valid)", type(arg.is_valid))
    used_labels = None
    if arg.dataset == "IN":
        X = scio.loadmat("data/Indian_pines_corrected.mat")["indian_pines_corrected"]
        y = scio.loadmat("data/Indian_pines_gt.mat")["indian_pines_gt"]
        VAL_SIZE = 1025
        used_labels = [1,2,4,7,9,10,11,13]
    elif arg.dataset == "PU":
        X = scio.loadmat("data/PaviaU.mat")["paviaU"]
        y = scio.loadmat("data/PaviaU_gt.mat")["paviaU_gt"]
        VAL_SIZE = 4281
    elif arg.dataset == "KSC":
        X = scio.loadmat("data/KSC.mat")["KSC"]
        y = scio.loadmat("data/KSC_gt.mat")["KSC_gt"]
    elif arg.dataset == "Salinas":
        X = scio.loadmat("data/Salinas_corrected.mat")["salinas_corrected"]
        y = scio.loadmat("data/Salinas_gt.mat")["salinas_gt"]
    elif arg.dataset == "Houston":
        X = scio.loadmat("data/houston15.mat")['data']
        mask_train = scio.loadmat("data/houston15_mask_train.mat")["mask_train"]
        mask_test = scio.loadmat("data/houston15_mask_test.mat")["mask_test"]


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    #X_train, y_train = oversampleWeakClasses(X_train, y_train)
    X = standartizeData(X)
    #X_train, y_train, X_test, y_test = build_data_v2(X, y)

    margin = 6
    X = padWithZeros(X, margin=margin)

    if arg.dataset == "Houston":
        num_classes = 15
    elif used_labels is not None:
        num_classes = len(used_labels)
    else:
        num_classes = len(np.unique(y)) - 1
    xshape = X.shape[1:]
    aoa = []
    aaa = []
    ak = []
    aapc = []
    for repterm in range(arg.repeat_term):

        if arg.dataset != "Houston":
            coords, labels = get_coordinates_labels(y)
            # train_coords, test_coords, train_labels, test_labels = train_test_split(coords, labels, test_size=arg.test_size)

            train_coords, train_labels, test_coords, test_labels = get_train_test(data=coords, data_labels=labels,
                                                                                  test_size=arg.test_size,
                                                                                  limited_num=arg.limited_num,
                                                                                  used_labels=used_labels)

        else:
            train_coords, train_labels = get_coordinates_labels(mask_train)
            test_coords, test_labels = get_coordinates_labels(mask_test)
        train_coords = train_coords + margin
        test_coords = test_coords + margin
        #X_test = Grammar(X, test_coords, method=arg.test_region)

        X_train = Grammar(X, train_coords, method="rect 11")
        y_train = train_labels
        y_test = test_labels

        if arg.data_augment:
            X_train, y_train = zmm_random_flip(X_train, y_train)  # rotation_and_flip(X_train, y_train)
            # X_train, y_train, X_test, y_test = build_data(X, y)
        X_train_shape = X_train.shape
        #X_test_shape = X_test.shape
        if len(X_train_shape) == 4:
            X_train = np.reshape(X_train, [X_train_shape[0], X_train_shape[1] * X_train_shape[2], X_train_shape[3]])
            #X_test = np.reshape(X_test, [X_test_shape[0], X_test_shape[1] * X_test_shape[2], X_test_shape[3]])

        #X_test = zeropad_to_max_len(X_test, max_len=arg.max_len)
        X_train = zeropad_to_max_len(X_train, max_len=arg.max_len)
        
        for i in range(num_classes):
            print("num train and test in class %d is %d / %d" % (i, (y_train == i).sum(), (y_test == i).sum()))
        #print("num_train", X_train.shape[0])
        #print("num_test", X_test.shape[0])
        print("num_classes", num_classes)

        train_generator = Data_Generator(X, y=y_train, use_coords=train_coords,
                                            batch_size=arg.batch_size,
                                            selection_rules=selection_rules,
                                            shuffle=True, till_end=False,
                                            max_len=arg.max_len)


        test_generator = Data_Generator(X, y=y_test, use_coords=test_coords,
                                        batch_size=1024,
                                        selection_rules=[arg.test_region]
                                        ,shuffle=False,
                                        till_end=True, max_len=arg.max_len)


        model = HSI_BERT(max_len = arg.max_len,
                         n_channel=xshape[-1],
                         max_depth=arg.max_depth,
                         num_head=arg.num_head,
                         num_hidden=arg.num_hidden,
                         drop_rate=arg.drop_rate,
                         attention_dropout=arg.attention_dropout,
                         num_classes = num_classes,
                         start_learning_rate=arg.start_learning_rate,
                         prembed=arg.prembed,
                         prembed_dim=arg.prembed_dim,
                         masking=arg.masking,
                         pooling=arg.pooling,
                         pool_size=arg.pool_size)

        model.build()

        print(arg)

        save_full_path = None

        if arg.save_model:
            if not os.path.exists(arg.save_path):
                os.mkdir(arg.save_path)
            save_full_path = arg.save_path+'/'+arg.dataset+"/model_%d_h%d_d%d"%(repterm, arg.num_head, arg.max_depth)+'/'+"model_%d_h%d_d%d.ckpt"%(repterm, arg.num_head, arg.max_depth)
            model_path = arg.save_path+'/'+arg.dataset+"/model_%d_h%d_d%d"%(repterm, arg.num_head, arg.max_depth)
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            np.save(os.path.join(model_path,"train_coords.npy"), train_coords - margin)
            np.save(os.path.join(model_path, "test_coords.npy"), test_coords - margin)
        #if arg.dataset == "Salinas":
        """
        print("Fitting generator")
        with timer("Fitting Generator Completed"):
            model.fit_generator(train_generator,
                            nb_epochs = arg.n_epochs,
                            log_every_n_samples = arg.log_every_n_samples,
                            save_path=save_full_path)

        #preds = model.predict_from_generator(test_generator)
        """
        print("Fitting normal data")
        with timer("Fitting Normal Data Completed"):
            model.fit(X_train, y_train, batch_size=arg.batch_size,
                      nb_epochs=arg.n_epochs,
                      log_every_n_samples=arg.log_every_n_samples,
                      save_path=save_full_path)

        with timer("Testing"):
            preds = model.predict_from_generator(test_generator)
        result = get_matrics(y_true=test_labels, y_pred=preds)
        oa = result['oa']
        aa = result["aa"]
        kappa = result["k"]
        apc = result["apc"]
        print("oa", oa)
        print('aa', aa)
        print("kappa", kappa)
        print("apc", apc.flatten())

        best_model = HSI_BERT(max_len = arg.max_len,
                         n_channel=xshape[-1],
                         max_depth=arg.max_depth,
                         num_head=arg.num_head,
                         num_hidden=arg.num_hidden,
                         drop_rate=arg.drop_rate,
                         attention_dropout=arg.attention_dropout,
                         num_classes = num_classes,
                         start_learning_rate=arg.start_learning_rate,
                         prembed=arg.prembed,
                         prembed_dim=arg.prembed_dim,
                         masking=arg.masking,
                         pooling=arg.pooling,
                         pool_size=arg.pool_size)
        best_model.restore(save_full_path)

        #if arg.dataset == "Salinas":
        #    preds = best_model.predict_from_generator(test_generator)
        #else:
        preds = best_model.predict_from_generator(test_generator)
        result = get_matrics(y_pred=preds, y_true=test_labels)
        oa = result['oa']
        aa = result["aa"]
        kappa = result["k"]
        apc = result["apc"]
        print("oa", oa)
        print('aa', aa)
        print("kappa", kappa)
        print("apc", apc.flatten())
        aoa.append(oa)
        aaa.append(aa)
        ak.append(kappa)
        aapc.append(apc)
        print(classification_report(test_labels, preds))
    aoa = np.array(aoa)
    aaa = np.array(aaa)
    ak = np.array(ak)
    std_aa = np.std(aaa)
    std_oa = np.std(aoa)
    std_ak = np.std(ak)
    aapc = np.concatenate(aapc, axis=0)
    print("mean oa", np.mean(aoa))
    print("std_oa", std_oa)
    print("mean aa", np.mean(aaa))
    print("std_aa", std_aa)
    print("mean kappa", np.mean(ak))
    print("sta_ak", std_ak)
    print("maapc", np.mean(aapc, axis=0))
    print("maapc_std", np.std(aapc, axis=0))
    print("below is aapc")
    print(aapc)

if __name__ =="__main__":
    main()
