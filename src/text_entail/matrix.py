# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 13:05:14 2014

@author: stevo
"""
from __future__ import print_function

import logging
import sys
import os
import cPickle

import numpy as np
from scipy.sparse import dok_matrix
from scipy.io import mmwrite, mmread
import text_entail.dictionary as td
import text_entail.io as tio

def w1Asfeature(d_triples, d_w1):
    """
    """
    w1_mat = dok_matrix((len(d_triples), len(d_triples._m2ids)))
    for w1, ids in d_triples._m2ids.items():
        j = d_w1.add(w1)
        for i in ids:
            w1_mat[i,j] = 1
    return w1_mat

def w2Asfeature(d_triples, d_w2):
    """
    """
    w2_mat = dok_matrix((len(d_triples), len(d_triples._r2ids)))
    for w2, ids in d_triples._r2ids.items():
        j = d_w2.add(w2)
        for i in ids:
            w2_mat[i,j] = 1
    return w2_mat

def ctxAsfeature(d_triples, d_ctx):
    """
    """
    ctx_mat = dok_matrix((len(d_triples), len(d_triples._l2ids)))
    for ctx, ids in d_triples._l2ids.items():
        j = d_ctx.add(ctx)
        for i in ids:
            ctx_mat[i,j] = 1
    return ctx_mat

def binarize_sparse_matrix(mat):
    """
    """
    mat = mat.astype(bool)
    mat = mat.astype(np.float64)
    return mat

def pred_vectors_with_context(preds_file, has_header=True):
    """
    """
    logging.info("creating predicate pairs class vector '{}'".format(preds_file))
    temp = []

    xy_predl_predr_entail = tio.read_preds_w_ctx(preds_file, has_header=has_header)

    d_triples = td.TripleDict() # rows
    duplicates = 0
    contradicting_duplicates = 0
    for ctx_X, ctx_Y, pred_l, pred_r, entailing in xy_predl_predr_entail:
        ctx = '{}\t{}'.format(ctx_X, ctx_Y)
        i = d_triples.add((ctx, pred_l, pred_r))
        if i < len(temp):
            label = 1 if entailing.strip().lower() == 'true' else 0
            print("omitting duplicate example: '{} {} {} {}' ".format(ctx, pred_l, pred_r, entailing) ,file=sys.stderr)
            duplicates += 1
            if temp[i] != label:
                print("duplicate example has different label: '{}' vs. '{}'".format(temp[i], label) ,file=sys.stderr)
                contradicting_duplicates += 1
        else:
            temp.append(1 if entailing.strip().lower() == 'true' else 0)
    vec = np.array(temp, dtype=np.float64)
    logging.info("finished creating arg pairs class vector '{}'".format(preds_file))
    logging.info("found {} duplicate examples with {} having contradicting labels.".format(duplicates, contradicting_duplicates))
    return vec, d_triples

def arg_l_arg_r_pairs_vector(args_file, file_contains_context=False, has_header=True):
    """
    """
    logging.info("creating arg pairs class vector '{}'".format(args_file))
    temp = []
    if file_contains_context:
        ctx_argl_argr_entail = tio.read_args_w_ctx(args_file, has_header=has_header)
    else:
        argl_argr_entail = tio.read_args_wo_ctx(args_file, has_header=has_header)
        def append_empty_context(tuples):
            for l,r,e in tuples:
                yield '', l, r, e
        ctx_argl_argr_entail = append_empty_context(argl_argr_entail)

    d_triples = td.TripleDict() # rows
    duplicates = 0
    contradicting_duplicates = 0
    for ctx, arg_l, arg_r, entailing in ctx_argl_argr_entail:
        i = d_triples.add((ctx, arg_l, arg_r))
        if i < len(temp):
            label = 1 if entailing.strip().lower() == 'true' else 0
            print("omitting duplicate example: '{} {} {} {}' ".format(ctx, arg_l, arg_r, entailing) ,file=sys.stderr)
            duplicates += 1
            if temp[i] != label:
                print("duplicate example has different label: '{}' vs. '{}'".format(temp[i], label) ,file=sys.stderr)
                contradicting_duplicates += 1
        else:
            temp.append(1 if entailing.strip().lower() == 'true' else 0)
    vec = np.array(temp, dtype=np.float64)
    logging.info("finished creating arg pairs class vector '{}'".format(args_file))
    logging.info("found {} duplicate examples with {} having contradicting labels.".format(duplicates, contradicting_duplicates))
    return vec, d_triples


def arg_l_arg_r_asjo_matrix(
    row_indices,
    jb_file,
    num_rows,
    col_indices,
    transform_w1 = lambda w1 : (w1[:w1.find('::@')], w1[w1.find('@::')+3:]),
    transform_w2sig = lambda w2sig : w2sig,
    mmfile_presuffix = '',
    reload = False):
    """
    """
    mm_file = os.path.splitext( jb_file )[0] + mmfile_presuffix + '.mm'
    if not reload:
        # legacy condition ( for files with file extension inside filename )
        if not os.path.exists(mm_file):
            mm_file = jb_file + mmfile_presuffix + '.mm'
        if os.path.exists(mm_file) and os.path.isfile(mm_file):
            logging.info("corresponding matrix file already exists for '{}'.".format(jb_file))
            logging.info("loading '{}'.".format(mm_file))
            mat = mmread(mm_file)
            with open(mm_file+'i','r') as f:
                col_indices._id2w = cPickle.load(f)
            for i, w in enumerate(col_indices._id2w):
                col_indices._w2id[w] = i
            logging.info("finished loading '{}'.".format(mm_file))
            return mat

    logging.info("creating arg pair feature matrix '{}'".format(jb_file))
    mat = dok_matrix((num_rows,1),dtype=np.float64) # len(d_pairs) = number of rows

    j_bs = tio.read_jb_file_filter_by_jo(jb_file, lambda jo : transform_w1(jo) in row_indices)
    for j, bs in j_bs:
        ks = row_indices[transform_w1(j)]
        for b, s in transform_w2sig(bs):
            l = col_indices.add(b)
            if mat.shape[1] <= l:
                mat.resize((mat.shape[0],l+1))
            for k in ks:
                mat[k,l] = float(s)
    logging.info("finished creating arg pair feature matrix '{}'".format(jb_file))
    logging.info("saving matrix to '{}'.".format(mm_file))
    with open(mm_file,'w') as f:
        mmwrite(f, mat)
    with open(mm_file+'i','w') as f:
        cPickle.dump(col_indices._id2w, f)
    logging.info("finshed saving matrix")
    return mat

def arg_asjo_matrix(
    row_indices,
    col_indices,
    jb_file,
    num_rows,
    transform_w1 = lambda w1 : w1,
    transform_w2sig = lambda w2sig : w2sig,
    mmfile_presuffix = '',
    reload = False):
    """
    """
    mm_file = os.path.splitext( jb_file )[0] + mmfile_presuffix + '.mm'
    if not reload:
        # legacy condition ( for files with file extension inside filename )
        if not os.path.exists(mm_file):
            mm_file = jb_file + mmfile_presuffix + '.mm'
        if os.path.exists(mm_file) and os.path.isfile(mm_file):
            logging.info("corresponding matrix file already exists for '{}'.".format(jb_file))
            logging.info("loading '{}'.".format(mm_file))
            mat = mmread(mm_file)
            with open(mm_file+'i','r') as f:
                col_indices._id2w = cPickle.load(f)
            for i, w in enumerate(col_indices._id2w):
                col_indices._w2id[w] = i
            logging.info("finished loading '{}'.".format(mm_file))
            return mat

    logging.info("creating arg feature matrix '{}'".format(jb_file))
    mat = dok_matrix((num_rows,1),dtype=np.float64) # number of rows x 1
    j_bs = tio.read_jb_file_filter_by_jo(jb_file, lambda jo : transform_w1(jo) in row_indices)
    for j, bs in j_bs:
        j = transform_w1(j)
        ks = row_indices[j]
        for b, s in transform_w2sig(bs):
            l = col_indices.add(b)
            if mat.shape[1] <= l:
                mat.resize((mat.shape[0],l+1))
            for k in ks:
                mat[k,l] = float(s)
    logging.info("finished creating arg feature matrix '{}'".format(jb_file))
    logging.info("saving matrix to '{}'.".format(mm_file))
    with open(mm_file,'w') as f:
        mmwrite(f, mat)
    with open(mm_file+'i','w') as f:
        cPickle.dump(col_indices._id2w, f)
    logging.info("finshed saving matrix")
    return mat

def arg_to_topic_matrix(
    args,
    word2topic_file,
    num_rows,
    transform_w = lambda w: w,
    mmfile_presuffix = '',
    reload = False):
    """
    """
    mm_file = os.path.splitext( word2topic_file )[0] + mmfile_presuffix + '.mm'
    if not reload:
        # legacy condition ( for files with file extension inside filename )
        if not os.path.exists(mm_file):
            mm_file = word2topic_file + mmfile_presuffix + '.mm'
        if os.path.exists(mm_file) and os.path.isfile(mm_file):
            logging.info("corresponding matrix file already exists for '{}'.".format(word2topic_file))
            logging.info("loading '{}'.".format(mm_file))
            mat = mmread(mm_file)
            logging.info("finished loading '{}'.".format(mm_file))
            return mat

    logging.info("creating topic feature matrix '{}'".format(word2topic_file))
    mat = dok_matrix((num_rows,1),dtype=np.float64) # number of rows x 1
    w2t = tio.read_word2topicfile(word2topic_file)
    for w, t in w2t:
        w = transform_w(w)
        if not w in args:
            continue
        ks = args[w]
        if mat.shape[1] <= t:
            mat.resize((mat.shape[0],t+1))
        for k in ks:
            mat[k,t] = 1
    logging.info("finished creating topic feature matrix '{}'".format(word2topic_file))

    logging.info("saving matrix to '{}'.".format(word2topic_file))
    with open(mm_file,'w') as f:
        mmwrite(f, mat)
    logging.info("finished saving matrix")
    return mat

def arg_l_arg_r_to_topic_matrix(
    row_indices,
    pair2topic_file,
    num_rows,
    transform_w = lambda w1 : (w1[:w1.find('::@')], w1[w1.find('@::')+3:]),
    mmfile_presuffix = '',
    reload = False):
    """
    """
    mm_file = os.path.splitext( pair2topic_file )[0] + mmfile_presuffix + '.mm'
    if not reload:
        # legacy condition ( for files with file extension inside filename )
        if not os.path.exists(mm_file):
            mm_file = pair2topic_file + mmfile_presuffix + '.mm'
        if os.path.exists(mm_file) and os.path.isfile(mm_file):
            logging.info("corresponding matrix file already exists for '{}'.".format(pair2topic_file))
            logging.info("loading '{}'.".format(mm_file))
            mat = mmread(mm_file)
            logging.info("finished loading '{}'.".format(mm_file))
            return mat

    logging.info("creating topic feature matrix '{}'".format(pair2topic_file))
    mat = dok_matrix((num_rows,1),dtype=np.float64) # number of rows x 1
    w2t = tio.read_word2topicfile(pair2topic_file)
    for w, t in w2t:
        p = transform_w(w)
        if p not in row_indices:
            continue
        ks = row_indices[p]
        if mat.shape[1] <= t:
            mat.resize((mat.shape[0],t+1))
        for k in ks:
            mat[k,t] = 1
    logging.info("finished creating topic feature matrix '{}'".format(pair2topic_file))

    logging.info("saving matrix to '{}'.".format(pair2topic_file))
    with open(mm_file,'w') as f:
        mmwrite(f, mat)
    logging.info("finished saving matrix")
    return mat

def topic_vector_matrix(
    row_indices,
    word2topicvector_file,
    num_rows,
    transform_w = lambda w: w,
    mmfile_presuffix = '',
    reload = False):
    """
    """
    mm_file = os.path.splitext(word2topicvector_file)[0] + mmfile_presuffix + '.mm'
    if not reload:
#        # legacy condition ( for files with file extension inside filename )
#        if not os.path.exists(mm_file):
#            mm_file = word2topic_file + mmfile_presuffix + '.mm'
        if os.path.exists(mm_file) and os.path.isfile(mm_file):
            logging.info("corresponding matrix file already exists for '{}'.".format(word2topicvector_file))
            logging.info("loading '{}'.".format(mm_file))
            mat = mmread(mm_file)
            logging.info("finished loading '{}'.".format(mm_file))
            return mat

    logging.info("creating topic vector feature matrix '{}'".format(word2topicvector_file))
    mat = dok_matrix((num_rows,1),dtype=np.float64) # number of rows x 1
    w2t = tio.read_word2topicvectorfile(word2topicvector_file)
    for w, t in w2t:
        w = transform_w(w)
        if not w in row_indices:
            continue
        t = np.array(t.split(' '), dtype=np.float)
        ks = row_indices[w]
        if mat.shape[1] < len(t):
            mat.resize((mat.shape[0],len(t)))
        for k in ks:
            mat[k,:] = t
    logging.info("finished creating topic feature matrix '{}'".format(word2topicvector_file))

    logging.info("saving matrix to '{}'.".format(word2topicvector_file))
    with open(mm_file,'w') as f:
        mmwrite(f, mat)
    logging.info("finished saving matrix")
    return mat