# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 09:43:23 2014

@author: stevo
@author: priska
"""

from __future__ import print_function
from __future__ import division

import logging

from sklearn.linear_model.logistic import LogisticRegression
import sklearn.metrics as metrics
import numpy as np
import random as rand

import text_entail.matrix as tm
import text_entail.io as tio

def run_baseline_classification_test(true_labels):
    """
    """
    #### create train and test split
    logging.info('preparing train and test set.')
    # always true baseline
    print('ALWAYS TRUE BASELINE:')
    predicted = np.ones(len(true_labels))
    calculate_statistics(true_labels, predicted)
    # always false baseline
    print('ALWAYS FALSE BASELINE:')
    true_labels_ = np.ones(len(true_labels)) - true_labels
    calculate_statistics(true_labels_, predicted)

def run_classification_test(mat, true_labels, binarize=True,
    percentage_train=0.8, print_train_test_set_stat = True,
    test_thresholds=False, random_seed=None, d_args=None, d_triples=None):
    """
    """
    ## binarize full matrix if desired
    if binarize:
        mat = tm.binarize_sparse_matrix(mat)

    #### create train and test split
    logging.info('preparing train and test set.')
    train_indexes, test_indexes = get_stratified_train_test_indexes(true_labels, percentage_train, random_seed)
    # train_indexes, test_indexes = get_train_test_indexes_presplit(d_args)
    # train_indexes, test_indexes = \
    #    get_train_test_indices_from_triples(d_triples, true_labels, percentage_train, random_seed)
    train_mat, test_mat, true_train_labels, true_test_labels = \
        split_matrix_to_train_and_test(mat, true_labels, train_indexes, test_indexes, print_stat=print_train_test_set_stat)
    model = classify(train_mat, test_mat, true_train_labels, true_test_labels, test_thresholds)
    return model

def get_stratified_train_test_indexes_notzero(mat, true_labels, percentage_train = 0.8, random_seed=None):
    """
    """
    r = rand.Random(x=random_seed)

    zero_idxes = np.squeeze(np.array(np.where(mat.sum(1) == 0)[0]))
    logging.info("number of samples with zero features: {} ({})".format(len(zero_idxes), len(zero_idxes) / len(true_labels)))

    true_labels_tmp = true_labels.copy();
    true_labels_tmp[zero_idxes] = -1;
    pos_idxes = np.where(true_labels_tmp == 1)[0]
    r.shuffle(pos_idxes)
    neg_idxes = np.where(true_labels_tmp == 0)[0]
    r.shuffle(neg_idxes)

    num_train_examples_pos = int(len(pos_idxes)*percentage_train)
    num_train_examples_neg = int(len(neg_idxes)*percentage_train)

    train_idxes = np.hstack((pos_idxes[:num_train_examples_pos], neg_idxes[:num_train_examples_neg]))
    test_idxes = np.hstack((pos_idxes[num_train_examples_pos:], neg_idxes[num_train_examples_neg:]))

    logging.info("size of training/test set: {}/{} ({})".format(len(train_idxes), len(test_idxes), len(train_idxes) / len(true_labels)))

    return train_idxes, test_idxes, zero_idxes

def get_stratified_train_test_indexes(true_labels, percentage_train = 0.8, random_seed=None):
    """
    """
    r = rand.Random(x=random_seed)
    pos_idxes = np.where(true_labels > 0)[0]
    r.shuffle(pos_idxes)
    neg_idxes = np.where(true_labels < 1)[0]
    r.shuffle(neg_idxes)

    num_train_examples_pos = int(len(pos_idxes)*percentage_train)
    num_train_examples_neg = int(len(neg_idxes)*percentage_train)

    train_idxes = np.hstack((pos_idxes[:num_train_examples_pos], neg_idxes[:num_train_examples_neg]))
    test_idxes = np.hstack((pos_idxes[num_train_examples_pos:], neg_idxes[num_train_examples_neg:]))

    return train_idxes, test_idxes

def get_fully_delex_train_test_indices_from_triples( d_triples, y_true,
    percentage_train_vocabulary=0.5, random_seed=None ):
    """
    Splits train and test set such as to maximize non-overlap of vocabulary;
    ie. the vocabulary of words is completely distinct in train and test set
    """

    v = set(d_triples._m2ids.keys()) | set(d_triples._r2ids.keys())
    v = list(v)

    r = rand.Random(x=random_seed)
    r.shuffle(v)

    num_train = int(len(v)*percentage_train_vocabulary)
    v_train = v[:num_train]
    v_test = v[num_train:]
    logging.info("size of training/test vocabulary: {}/{} ({})".format(len(v_train), len(v)-len(v_train), len(v_train) / len(v)))

    idx_train = list()
    idx_test = list()
    idx_zero = list()
    filtered_by_delex = list()
    for i, (_, w1, w2) in enumerate(d_triples._id2triple):
        if w1 in v_train and w2 in v_train:
            idx_train.append(i);
        elif w1 in v_test and w2 in v_test:
            idx_test.append(i)
        else:
            filtered_by_delex.append(i)

    usable_dataset_size = len(idx_train) + len(idx_test)
    logging.info("number of samples with zero features: {} ({})".format(len(idx_zero), len(idx_zero) / len(y_true)))
    logging.info("number of samples filtered by delexicalization: {} ({})".format(len(filtered_by_delex), len(filtered_by_delex) / len(y_true)))
    logging.info("number of usable samples: {}/{} ({})".format(usable_dataset_size, len(y_true), usable_dataset_size / len(y_true)))
    logging.info("size of training|test set: {}|{} ({})".format(len(idx_train), len(idx_test), usable_dataset_size))

    return np.array(idx_train), np.array(idx_test)

def get_fully_delex_train_test_indices_from_triples_notzero(d_triples, mat, y_true,
    percentage_train_vocabulary=0.5, random_seed=None ):
    """
    Splits train and test set such as to maximize non-overlap of vocabulary;
    ie. the vocabulary of words is completely distinct in train and test set
    """

    v = set(d_triples._m2ids.keys()) | set(d_triples._r2ids.keys())
    v = list(v)

    r = rand.Random(x=random_seed)
    r.shuffle(v)

    num_train = int(len(v)*percentage_train_vocabulary)
    v_train = v[:num_train]
    v_test = v[num_train:]
    logging.info("size of training/test vocabulary: {}/{} ({})".format(len(v_train), len(v)-len(v_train), len(v_train) / len(v)))

    idx_train = list()
    idx_test = list()
    idx_zero = list()
    filtered_by_delex = list()
    for i, (_, w1, w2) in enumerate(d_triples._id2triple):
        if mat[i,:].sum() == 0:
            idx_zero.append(i)
            continue
        if w1 in v_train and w2 in v_train:
            idx_train.append(i);
        elif w1 in v_test and w2 in v_test:
            idx_test.append(i)
        else:
            filtered_by_delex.append(i)

    usable_dataset_size = len(idx_train) + len(idx_test)
    logging.info("number of samples with zero features: {} ({})".format(len(idx_zero), len(idx_zero) / len(y_true)))
    logging.info("number of samples filtered by delexicalization: {} ({})".format(len(filtered_by_delex), len(filtered_by_delex) / len(y_true)))
    logging.info("number of usable samples: {}/{} ({})".format(usable_dataset_size, len(y_true), usable_dataset_size / len(y_true)))
    logging.info("size of training|test set: {}|{} ({})".format(len(idx_train), len(idx_test), usable_dataset_size))

    return np.array(idx_train), np.array(idx_test), np.array(idx_zero)

def get_train_test_indices_from_triples( d_triples, y_true, percentage_train=0.8,
    random_seed=None ):
    """
    Splits train and test set such as to maximize non-overlap of vocabulary;
    ie. allows a right lexeme to only be either in the train or in the test set.
    """
    # get right words
    w2 = d_triples._r2ids.keys()

    # for every right word collect the ids of the triples it occurs in
    w2idx = d_triples._r2ids

    # shuffle w2
    r = rand.Random( x=random_seed )
    r.shuffle( w2 )

    #TODO: the split into train and test data does not yet consider classes

    # split right words into train and test set according to percentage_train
    split_num = int( len( w2 ) * percentage_train )
    w2_train = w2[:split_num]
    w2_test  = w2[split_num:]

    idx_train = []
    for w in w2_train:
        idx_train.extend( w2idx[w] )

    idx_test  = []
    for w in w2_test:
        idx_test.extend( w2idx[w] )

    # return respective ids as train and test indices
    return np.asarray( idx_train ), np.asarray( idx_test )


def get_train_test_indexes_presplit(d_triples):
    """
    """
    t1 = tio.read_args_w_ctx('../data/updates/4/args_v2_am.tsv', has_header=False)
    train_ids = []
    for ctx, arg_l, arg_r, __ in t1:
        id_ = d_triples.get_triple_id((ctx, arg_l, arg_r))
        train_ids.append(id_)

    t2 = tio.read_args_w_ctx('../data/updates/4/args_v2_nz.tsv', has_header=False)
    test_ids = []
    for ctx, arg_l, arg_r, __ in t2:
        id_ = d_triples.get_triple_id((ctx, arg_l, arg_r))
        test_ids.append(id_)

    return train_ids, test_ids


def split_matrix_to_train_and_test(mat, true_labels, train_indexes, test_indexes,
    print_stat=False):
    """
    """
    train_mat = mat[train_indexes,:]
    true_train_labels = true_labels[train_indexes]
    test_mat = mat[test_indexes,:]
    true_test_labels = true_labels[test_indexes]

    if print_stat:
        print('======')
        print('  percentage of training examples: {}\n  num training examples: \
            {} ({}/{})\n  num testing examples:  {} ({}/{})'.format(\
            len(train_indexes) / len(true_labels),\
            len(train_indexes),\
            len(np.where(true_train_labels > 0)[0]),\
            len(np.where(true_train_labels < 1)[0]),\
            len(test_indexes),\
            len(np.where(true_test_labels > 0)[0]),\
            len(np.where(true_test_labels < 1)[0])))
        print('======')

    return train_mat, test_mat, true_train_labels, true_test_labels

def clazzify(train_mat, test_mat, true_train_labels):
    """
    """
    # learn
    logging.info('learning...')
    model = LogisticRegression(random_state=17, penalty='l1')
    model.fit(train_mat, true_train_labels)
    logging.info('finished learning.')

    # test
    logging.info('testing')
    predicted_test_labels = model.predict(test_mat)
    logging.info('finished testing')

    return predicted_test_labels, model

def classify(train_mat, test_mat, true_train_labels, true_test_labels,
    test_thresholds=False):
    """
    """
    predicted_test_labels, model = clazzify(train_mat, test_mat, true_train_labels)
    calculate_statistics(true_test_labels, predicted_test_labels)

    ###################
    ## test
    if test_thresholds:
        logging.info('testing thresholds')
        predicted_test_labels = model.predict_proba(test_mat)

        # Just take the probability of the positive classes
        predicted_test_labels = predicted_test_labels[:, 1]

        # From here on, it's just precision-recall calculations
        n = 10
        p_error = np.zeros(n+2)
        n_error = np.zeros(n+2)
        t = np.array([-0.05 + float(i)/n for i in xrange(n+2)])

        for i, e in enumerate(true_test_labels):
            prediction = predicted_test_labels[i]
            if e==1:
                p_error += ( np.sign(t - prediction) + 1)/2
                # p_error += (-np.sign(t - prediction) + 1)/2
            else:
                n_error += (-np.sign(t - prediction) + 1)/2
                # n_error += ( np.sign(t - prediction) + 1)/2

        total_p = true_test_labels.sum()
        total_n = true_test_labels.shape[0] - total_p
        print('Number of positive examples:', total_p)
        print('Number of negative examples:', total_n)

        tp = total_p - p_error
        fn = p_error
        fp = n_error
        tn = total_n - n_error

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)

        print('TP:', tp)
        print('FN:', fn)
        print('FP:', fp)
        print('TN:', tn)

        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('Best:', np.nanmax(f1))

    return model

def calculate_statistics(true_test_labels, predicted_test_labels):
    """
    """
    conf_matrix = metrics.confusion_matrix(true_test_labels, predicted_test_labels, labels=[1,0])
    accuracy = metrics.accuracy_score(true_test_labels, predicted_test_labels)
    avg_precision = metrics.average_precision_score( true_test_labels, predicted_test_labels )

    print('======')
    print()
    print('   TP: {}\n   FP: {}\n   FN: {}\n   TN: {}\n'.format(conf_matrix[0,0],
        conf_matrix[1,0], conf_matrix[0,1], conf_matrix[1,1]))
    print('  Acc: {}'.format(accuracy))
    print('   AP: {}'.format(avg_precision))
    print()
    # print('  Pr(1): {}\n  Re(1): {}\n  F1(1): {}\n'.format(prec_1, rec_1, f1_1))
    # print('  Pr(0): {}\n  Re(0): {}\n  F1(0): {}\n'.format(prec_0, rec_0, f1_0))
    # print('  Pr_w(0.5): {}\n  Re_w(0.5): {}\n  F1_w(0.5): {}\n'.format(prec, rec, f1))
    # print('  Pr_w(c): {}\n  Re_w(c): {}\n  F1_w(c): {}'.format(prec_w, rec_w, f1_w))

    # compute classification report ( incl. precision, recall, f_score etc. )
    report = metrics.classification_report( true_test_labels, predicted_test_labels )
    print( report )
    print('======')
