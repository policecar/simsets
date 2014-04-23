# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:58:13 2014

@author: stevo
"""

from __future__ import print_function;
from __future__ import division;
import logging;
import text_entail.matrix as tm;
import text_entail.dictionary as td;
import text_entail.classify as tc;
import scipy.sparse as sparse;
import resource_files as rf;
import numpy as np;

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG);

transform_predicate = lambda w1 : 'Y {} X'.format(w1[:-3]).replace('_', ' ') if w1.endswith('^-1') else 'X {} Y'.format(w1.replace('_', ' '));

logging.info('loading true labels');
true_labels, d_triples = tm.pred_vectors_with_context(\
    rf.local_preds_w_ctx, has_header=True);

## context features
logging.info('loading predicate context matrices');
d_ctx = td.Dict();
mat_pred_l_ctx = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_ctx,
    rf.local_nodes_counts,\
    len(d_triples),\
    transform_w1 = transform_predicate, \
    mmfile_presuffix='.predsctx.ctx_pred_l', reload=False);
 
mat_pred_r_ctx = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_ctx,
    rf.local_nodes_counts,\
    len(d_triples),\
    transform_w1 = transform_predicate, \
    mmfile_presuffix='.predsctx.ctx_pred_r', reload=False);

## create some extra matrices
logging.info('creating argument context intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_pred_l_ctx.shape[1] < mat_pred_r_ctx.shape[1]:
    if sparse.isspmatrix_coo(mat_pred_l_ctx):
        mat_pred_l_ctx = mat_pred_l_ctx.todok();
    mat_pred_l_ctx.resize(mat_pred_r_ctx.shape);
if mat_pred_r_ctx.shape[1] < mat_pred_l_ctx.shape[1]:
    if sparse.isspmatrix_coo(mat_pred_r_ctx):
        mat_pred_r_ctx = mat_pred_r_ctx.todok();
    mat_pred_r_ctx.resize(mat_pred_l_ctx.shape);

if not sparse.isspmatrix_coo(mat_pred_l_ctx):
    mat_pred_l_ctx = mat_pred_l_ctx.tocoo();
if not sparse.isspmatrix_coo(mat_pred_r_ctx):
    mat_pred_r_ctx = mat_pred_r_ctx.tocoo();

mat_pred_l_ctx = mat_pred_l_ctx.astype(bool);
mat_pred_r_ctx = mat_pred_r_ctx.astype(bool);

mat_pred_union_ctx = mat_pred_l_ctx + mat_pred_r_ctx;
mat_pred_diff_ctx = mat_pred_l_ctx != mat_pred_r_ctx;
mat_pred_inters_ctx = mat_pred_union_ctx - mat_pred_diff_ctx;
mat_pred_l_minus_r_ctx = mat_pred_union_ctx - mat_pred_r_ctx;
mat_pred_r_minus_l_ctx = mat_pred_union_ctx - mat_pred_l_ctx;

###
## topic features
logging.info('loading lda feature matrices.')
mat_pred_l_topic = tm.arg_to_topic_matrix(d_triples._m2ids,\
    rf.local_svo_lda_w2t,\
    len(d_triples),\
    transform_w = transform_predicate, \
    mmfile_presuffix='.predsctx.topic_pred_l', reload=False);

mat_pred_r_topic = tm.arg_to_topic_matrix(d_triples._r2ids,\
    rf.local_svo_lda_w2t,\
    len(d_triples),\
    transform_w = transform_predicate, \
    mmfile_presuffix='.predsctx.topic_pred_r', reload=False);

###
## distributionally similar args for each arg
logging.info('loading similar arguments.')
d_sim_pred = td.Dict();
mat_sim_pred_l = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_sim_pred,
    rf.local_nodes_dt,\
    len(d_triples),\
    transform_w1 = transform_predicate, \
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.predsctx.sim_arg_l', reload=False);

mat_sim_pred_r = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_sim_pred,
    rf.local_nodes_dt,\
    len(d_triples),\
    transform_w1 = transform_predicate, \
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.predsctx.sim_arg_r', reload=False);

## create some extra matrices
logging.info('creating similar arguments intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_sim_pred_l.shape[1] < mat_sim_pred_r.shape[1]:
    if sparse.isspmatrix_coo(mat_sim_pred_l):
        mat_sim_pred_l = mat_sim_pred_l.todok();
    mat_sim_pred_l.resize(mat_sim_pred_r.shape);
if mat_sim_pred_r.shape[1] < mat_sim_pred_l.shape[1]:
    if sparse.isspmatrix_coo(mat_sim_pred_r):
        mat_sim_pred_r = mat_sim_pred_r.todok();
    mat_sim_pred_r.resize(mat_sim_pred_l.shape);    

if not sparse.isspmatrix_coo(mat_sim_pred_l):
    mat_sim_pred_l = mat_sim_pred_l.tocoo();
if not sparse.isspmatrix_coo(mat_sim_pred_r):
    mat_sim_pred_r = mat_sim_pred_r.tocoo();
    
mat_sim_pred_l = mat_sim_pred_l.astype(bool);
mat_sim_pred_r = mat_sim_pred_r.astype(bool);

mat_sim_union_pred = mat_sim_pred_l + mat_sim_pred_r;
mat_sim_diff_pred = mat_sim_pred_l != mat_sim_pred_r;
mat_sim_inters_pred = mat_sim_union_pred - mat_sim_diff_pred;
mat_sim_l_minus_r_pred = mat_sim_union_pred - mat_sim_pred_r;
mat_sim_r_minus_l_pred = mat_sim_union_pred - mat_sim_pred_l;

### ### ### ###
  ### ### ### ###
### ### ### ###

logging.info('stacking matrices.');
# stack only two matrices at a time because of memory issues
mat = sparse.csr_matrix((len(d_triples),1),dtype=np.float64); #  initialize with (num_rows x 1) matrix, a dummy vector because scipy.sparse does not allow to stack empty matrices

# context feat. of pred_l and pred_r individually
mat = sparse.hstack((mat, mat_pred_l_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_r_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_union_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_diff_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_inters_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_l_minus_r_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_pred_r_minus_l_ctx.astype(np.float64))).tocsr();

# topic feat. of arg_l and arg_r
mat = sparse.hstack((mat, mat_pred_l_topic));
mat = sparse.hstack((mat, mat_pred_r_topic)).tocsr();

# similarity feat. of arg_l and arg_r individually
mat = sparse.hstack((mat, mat_sim_pred_l.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_pred_r.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_union_pred.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_diff_pred.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_inters_pred.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_l_minus_r_pred.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_r_minus_l_pred.astype(np.float64))).tocsr();

mat = mat.tocsr()[:,1:]; # remove the first dummy vector

tc.run_classification_test(mat, true_labels, binarize=True, percentage_train=0.8, print_train_test_set_stat=True, test_thresholds=False, random_seed=623519);