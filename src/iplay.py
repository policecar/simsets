#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:44:30 2014

@author: stevo
"""
from __future__ import print_function;
import text_entail.classify as tc;
import text_entail.matrix as tm;
import logging;
import scipy.sparse as sparse;
import re;
import numpy as np;

## change dataset here
# import ent_args_dataset  as clc;
# import ent_args_ctx_dataset  as clc;
import bless_dataset  as clc;

reload(logging); logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG);

_range_pattern = re.compile('\s?(?P<from>\d+)\s*-\s*(?P<to>\d+)\s?');

def readFirstInput(w1):
    while True:
        _in = raw_input("Enter a term (type <Enter> for previous term, type ? to view available terms, type cs! for random stratified classification (80/20), type cd! for delex classification (80/20), type q! to quit): ");
        if _in == '':
            _in = w1;
        if _in == 'q!':
            raise KeyboardInterrupt();
        if _in == 'cs!':
            return _in;
        if _in == 'cd!':
            return _in;
        if _in == '?':
            for c in range(0,len(_v),20):
                print('\n'.join(_v[c:c+20]));
                if c+20 < len(_v):
                    _inn = raw_input('Enter term or type <Enter> for more: ');
                    if len(_inn.strip()) > 0:
                        _in = _inn;
                        break;
            if _in == '?':
                continue;
        if _in in _v:
            return _in;
        print("'{}' not in vocabulary, please enter a another term.".format(_in));
        _in = None;

def readSecondInput(w1, w2):
    v2 = sorted(set([_d_triples.get_triple(i)[2] for i in _d_triples.get_middle_element_ids(w1)]));
    while True:
        _in = raw_input("Enter a second term (type <Enter> for previous term, type ? to view available terms, type q! to quit): ");
        if _in == '':
            _in = w2;
        if _in == 'q!':
            raise KeyboardInterrupt();
        if _in == '?':
            for c in range(0,len(v2),20):
                print('\n'.join(v2[c:c+20]));
                if c+20 < len(v2):
                    _inn = raw_input('Enter term or type <Enter> for more: ');
                    if len(_inn.strip()):
                        _in = _inn;
                        break;
            if _in == '?':
                continue;
        if _in in v2:
            return _in;
        print("'{}' not in vocabulary, please enter a another term.".format(_in));
        _in = None;


## main
_d_triples, true_labels = clc.load_classification_data();
matrices = clc.load_matrices(_d_triples);

_v = sorted(set(_d_triples._m2ids.keys()));

_w1 = None;
_w2 = '';
_f = [0,];
try:
    while True:
        _w1 = readFirstInput(_w1);
        if not '!' in _w1:
            print('first word is: ' + _w1);
            _w2 = readSecondInput(_w1, _w2);
            print('second word is: ' + _w2);

        while True:
            print('Which feature sets do you want to use for classification?');
            print('\n'.join(['{}: {}'.format(i, x) for i,(x,_,_) in enumerate(matrices)]));
            _in = raw_input('Enter space separated numbers (type <Enter> to use previous feature set, type a! for all feature sets, type q! to quit): ');
            try:
                if not _in.strip():
                    _in = ' '.join([str(f) for f in _f]);
                if _in == 'q!':
                    raise KeyboardInterrupt();
                if 'a' in _in:
                    _in = ' '.join([str(f) for f in range(len(matrices))]);

                _new_in = _in;
                m = _range_pattern.search(_in);
                while m:
                    f = m.group('from');
                    t = m.group('to');
                    print('{} {}'.format(f,t))
                    l = [str(i) for i in range(int(f),int(t)+1)];
                    print('{}'.format(l));
                    _new_in = _new_in.replace(m.group(), ' '.join(l));
                    m = _range_pattern.search(_in, m.end());
                _in = _new_in;
                _f = [int(x) for x in _in.split(' ')];
                break;
            except Exception as e:
                print('Something went wrong, please try again. ({}: {})'.format(type(e), e.message));

        print('Using feature matrices: {}'.format(_f));

        # stack
        logging.info('stacking matrices.');
        _mat = sparse.csr_matrix((len(_d_triples),1));
        _colheader = [];
        for i in _f:
            matrixdescr, matrix, d = matrices[i];
            logging.info('stacking matrix \'{}\''.format(matrixdescr));
            _mat = sparse.hstack((_mat, matrix));
            if d:
                _colheader += ['[FS{} {}]: {}'.format(i, matrixdescr, x) for x in d._id2w];
            else:
                _colheader += ['[FS{} {}]: f_{}'.format(i, matrixdescr, x) for x in range(matrix.shape[1])];
        _colheader = np.array(_colheader);
        _mat = _mat.tocsr()[:,1:];

        if not '!' in _w1:
            w1w2_idxs = _d_triples.get_right_tuple_ids((_w1,_w2));
            _train_idxes = np.delete(np.arange(0,len(_d_triples)), w1w2_idxs);
            _test_idxes = w1w2_idxs;
            print('Testing Context - ArgL - ArgR triples: \n{}'.format('\n'.join(['{} - {}'.format(i, _d_triples.get_triple(i)) for i in w1w2_idxs])));
        else:
            if 's' in _w1:
                _train_idxes, _test_idxes, _zero_v_idxes =  tc.get_stratified_train_test_indexes_notzero(_mat,true_labels, percentage_train=0.8, random_seed=623519);
#                _test_idxes = np.hstack((_test_idxes, _zero_v_idxes));
            if 'd' in _w1:
                _train_idxes, _test_idxes, _zero_v_idxes =  tc.get_fully_delex_train_test_indices_from_triples_notzero(_d_triples, _mat, true_labels, percentage_train_vocabulary=0.5, random_seed=623519);
#                _test_idxes = np.hstack((_test_idxes, _zero_v_idxes));

        _in = raw_input('Binarize feature matrix? ([{}]es, [n]o, type q! to quit): '.format('\033[4m\033[1my\033[0m'));
        if _in == 'q!':
            raise KeyboardInterrupt();
        if not _in.strip() or _in.strip().lower() == 'y':
            _mat = tm.binarize_sparse_matrix(_mat);

        _mat_train = _mat[_train_idxes,:];
        _train_labels = true_labels[_train_idxes];
        _mat_test = _mat[_test_idxes,:];
        _test_labels = true_labels[_test_idxes];

        predicted_test_labels, model = tc.clazzify(_mat_train, _mat_test, _train_labels);
        sorted_idxs = np.argsort(np.abs(model.coef_[0]))[::-1]; # sort and reverse indices, model.coef_ is just a (1 x n) matrix
        print('Coefficients:\n\t{}\n\t{}'.format(model.intercept_[0], '\n\t'.join(['{:+.3f} {:6d} {}'.format(model.coef_[0][i], i, _colheader[i]) for i in sorted_idxs[:20]])));

        _in = raw_input('Enter y to predict {} zero-vector(s) with default class (0) (press <Enter> or n to not classify zero-vectors, type q! to quit): '.format(len(_zero_v_idxes)));
        if _in == 'q!':
            raise KeyboardInterrupt();
        if _in.strip().lower() == 'y' and len(_zero_v_idxes) > 0:
            _test_idxes = np.hstack((_test_idxes, _zero_v_idxes));
            _test_labels = np.hstack((_test_labels, true_labels[_zero_v_idxes]));
            predicted_test_labels = np.hstack((predicted_test_labels, np.zeros(len(_zero_v_idxes))));

        if '!' in _w1:
            tc.calculate_statistics(_test_labels, predicted_test_labels)
        else:
            coef_samples = _mat_test.multiply(model.coef_).A;
            for i in range(coef_samples.shape[0]):
                sorted_idxs = np.argsort(np.abs(np.array(coef_samples[i])))[::-1]; # sort and reverse indices, model.coef_ is just a (1 x n) matrix
                print('Coefficients {} (predicted: {}, real: {}):\n\t{}'.format(_d_triples.get_triple(_test_idxes[i]), predicted_test_labels[i], _test_labels[i], '\n\t'.join(['{:+.3f} {:6d} {}'.format(coef_samples[i][j], j, _colheader[j]) for j in sorted_idxs[:20]])));

except KeyboardInterrupt:
    pass;
