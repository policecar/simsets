# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:15:26 2014

@author: stevo
"""

from __future__ import print_function;
from __future__ import division;
import logging;
import text_entail.matrix as tm;
import text_entail.dictionary as td;
import text_entail.classify as tc;
import text_entail.util as tu;
import scipy.sparse as sparse;
import resource_files as rf;
import numpy as np;

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG);

logging.info('loading true labels');
true_labels, d_triples = tm.arg_l_arg_r_pairs_vector(\
    rf.local_args_w_ctx, file_contains_context=True, has_header=True);

## relation pair features
logging.info('loading paths between argument pairs');
d_paths = td.Dict();
mat_paths = tm.arg_l_arg_r_asjo_matrix(d_triples._rtuple2ids, \
    rf.local_nodes_flipped_counts,\
    len(d_triples),
    col_indices = d_paths, \
    mmfile_presuffix='.argsctx.paths', reload=False);

logging.info('loading similar argument pairs');
d_sim_pairs = td.Dict();
mat_sim_pairs = tm.arg_l_arg_r_asjo_matrix(d_triples._rtuple2ids, \
    rf.local_nodes_flipped_dt,\
    len(d_triples),
    col_indices = d_sim_pairs, \
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.argsctx.simpairs', reload=False);

## context features
logging.info('loading argument context matrices');
d_ctx = td.Dict();
mat_arg_l_ctx = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_ctx,
    rf.local_nodes_counts,\
    len(d_triples),\
    mmfile_presuffix='.argsctx.ctx_arg_l', reload=False);
 
mat_arg_r_ctx = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_ctx,
    rf.local_nodes_counts,\
    len(d_triples),\
    mmfile_presuffix='.argsctx.ctx_arg_r', reload=False);

## create some extra matrices
logging.info('creating argument context intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_arg_l_ctx.shape[1] < mat_arg_r_ctx.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_l_ctx):
        mat_arg_l_ctx = mat_arg_l_ctx.todok();
    mat_arg_l_ctx.resize(mat_arg_r_ctx.shape);
if mat_arg_r_ctx.shape[1] < mat_arg_l_ctx.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_r_ctx):
        mat_arg_r_ctx = mat_arg_r_ctx.todok();
    mat_arg_r_ctx.resize(mat_arg_l_ctx.shape);

if not sparse.isspmatrix_coo(mat_arg_l_ctx):
    mat_arg_l_ctx = mat_arg_l_ctx.tocoo();
if not sparse.isspmatrix_coo(mat_arg_r_ctx):
    mat_arg_r_ctx = mat_arg_r_ctx.tocoo();

mat_arg_l_ctx = mat_arg_l_ctx.astype(bool);
mat_arg_r_ctx = mat_arg_r_ctx.astype(bool);

mat_arg_union_ctx = mat_arg_l_ctx + mat_arg_r_ctx;
mat_arg_diff_ctx = mat_arg_l_ctx != mat_arg_r_ctx;
mat_arg_inters_ctx = mat_arg_union_ctx - mat_arg_diff_ctx;
mat_arg_l_minus_r_ctx = mat_arg_union_ctx - mat_arg_r_ctx;
mat_arg_r_minus_l_ctx = mat_arg_union_ctx - mat_arg_l_ctx;

## topic features
logging.info('loading lda feature matrices.')
logging.info('loading lda feature matrices.')
mat_topic = tm.arg_l_arg_r_to_topic_matrix(d_triples._rtuple2ids,\
    rf.local_svo_flipped_lda_w2t,\
    len(d_triples), \
    mmfile_presuffix='.argsctx.topic_pairs', reload=False);
    
mat_arg_l_topic = tm.arg_to_topic_matrix(d_triples._m2ids,\
    rf.local_svo_lda_w2t,\
    len(d_triples),\
    mmfile_presuffix='.argsctx.topic_arg_l', reload=False);

mat_arg_r_topic = tm.arg_to_topic_matrix(d_triples._r2ids,\
    rf.local_svo_lda_w2t,\
    len(d_triples),\
    mmfile_presuffix='.argsctx.topic_arg_r', reload=False);

# distributionally similar args for each arg
logging.info('loading similar arguments.')
d_arg = td.Dict();
mat_sim_arg_l = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_arg,
    rf.local_nodes_dt,\
    len(d_triples),\
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.argsctx.sim_arg_l', reload=False);

mat_sim_arg_r = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_arg,
    rf.local_nodes_dt,\
    len(d_triples),\
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.argsctx.sim_arg_r', reload=False);

## create some extra matrices
logging.info('creating similar arguments intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_sim_arg_l.shape[1] < mat_sim_arg_r.shape[1]:
    if sparse.isspmatrix_coo(mat_sim_arg_l):
        mat_sim_arg_l = mat_sim_arg_l.todok();
    mat_sim_arg_l.resize(mat_sim_arg_r.shape);
if mat_sim_arg_r.shape[1] < mat_sim_arg_l.shape[1]:
    if sparse.isspmatrix_coo(mat_sim_arg_r):
        mat_sim_arg_r = mat_sim_arg_r.todok();
    mat_sim_arg_r.resize(mat_sim_arg_l.shape);    

if not sparse.isspmatrix_coo(mat_sim_arg_l):
    mat_sim_arg_l = mat_sim_arg_l.tocoo();
if not sparse.isspmatrix_coo(mat_sim_arg_r):
    mat_sim_arg_r = mat_sim_arg_r.tocoo();
    
mat_sim_arg_l = mat_sim_arg_l.astype(bool);
mat_sim_arg_r = mat_sim_arg_r.astype(bool);

mat_sim_union_arg = mat_sim_arg_l + mat_sim_arg_r;
mat_sim_diff_arg = mat_sim_arg_l != mat_sim_arg_r;
mat_sim_inters_arg = mat_sim_union_arg - mat_sim_diff_arg;
mat_sim_l_minus_r_arg = mat_sim_union_arg - mat_sim_arg_r;
mat_sim_r_minus_l_arg = mat_sim_union_arg - mat_sim_arg_l;

# context sensitive features
logging.info('loading paths between proposition contexts and argument pairs');
l_ctx_tuple2ids = {};
r_ctx_tuple2ids = {};
for i, (svo_proposition, arg_l, arg_r) in enumerate(d_triples._id2triple):
    ctx = tu.get_lhs_arg(svo_proposition);
    if 'X' == ctx:
        ctx = tu.get_rhs_arg(svo_proposition);
    l_ctx_tuple = (ctx,arg_l);
    r_ctx_tuple = (ctx,arg_r);
    if l_ctx_tuple not in l_ctx_tuple2ids:
        l_ctx_tuple2ids[l_ctx_tuple] = [];
    if r_ctx_tuple not in r_ctx_tuple2ids:
        r_ctx_tuple2ids[r_ctx_tuple] = [];
    l_ctx_tuple2ids[l_ctx_tuple].append(i);
    r_ctx_tuple2ids[r_ctx_tuple].append(i);

d_paths_ctx = td.Dict();
mat_arg_l_ctx_paths = tm.arg_l_arg_r_asjo_matrix(l_ctx_tuple2ids, \
    rf.local_nodes_flipped_counts,\
    len(d_triples),
    col_indices = d_paths_ctx, \
    mmfile_presuffix='.argsctx.paths_ctx_argl', reload=False);
    
mat_arg_r_ctx_paths = tm.arg_l_arg_r_asjo_matrix(r_ctx_tuple2ids, \
    rf.local_nodes_flipped_counts,\
    len(d_triples),
    col_indices = d_paths_ctx, \
    mmfile_presuffix='.argsctx.paths_ctx_argr', reload=False);

logging.info('creating proposition context intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_arg_l_ctx_paths.shape[1] < mat_arg_r_ctx_paths.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_l_ctx_paths):
        mat_arg_l_ctx_paths = mat_arg_l_ctx_paths.todok();
    mat_arg_l_ctx_paths.resize(mat_arg_r_ctx_paths.shape);
if mat_arg_r_ctx_paths.shape[1] < mat_arg_l_ctx_paths.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_r_ctx_paths):
        mat_arg_r_ctx_paths = mat_arg_r_ctx_paths.todok();
    mat_arg_r_ctx_paths.resize(mat_arg_l_ctx_paths.shape);    

if not sparse.isspmatrix_coo(mat_arg_l_ctx_paths):
    mat_arg_l_ctx_paths = mat_arg_l_ctx_paths.tocoo();
if not sparse.isspmatrix_coo(mat_arg_r_ctx_paths):
    mat_arg_r_ctx_paths = mat_arg_r_ctx_paths.tocoo();

mat_arg_l_ctx_paths = mat_arg_l_ctx_paths.astype(bool);
mat_arg_r_ctx_paths = mat_arg_r_ctx_paths.astype(bool);
mat_arg_ctx_union_arg = mat_arg_l_ctx_paths + mat_arg_r_ctx_paths;
mat_arg_ctx_diff_arg = mat_arg_l_ctx_paths != mat_arg_r_ctx_paths;
mat_arg_ctx_inters_arg = mat_arg_ctx_union_arg - mat_arg_ctx_diff_arg;
mat_arg_ctx_l_minus_r_arg = mat_arg_ctx_union_arg - mat_arg_r_ctx_paths;
mat_arg_ctx_r_minus_l_arg = mat_arg_ctx_union_arg - mat_arg_l_ctx_paths;

# similar proposition contexts
logging.info('loading similar proposition context argument pairs');
d_sim_paths_ctx = td.Dict();
mat_arg_l_ctx_sim_paths = tm.arg_l_arg_r_asjo_matrix(l_ctx_tuple2ids, \
    rf.local_nodes_flipped_dt,\
    len(d_triples),
    col_indices = d_sim_paths_ctx, \
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.argsctx.sim_paths_ctx_argl', reload=False);
    
mat_arg_r_ctx_sim_paths = tm.arg_l_arg_r_asjo_matrix(r_ctx_tuple2ids, \
    rf.local_nodes_flipped_dt,\
    len(d_triples),
    col_indices = d_sim_paths_ctx, \
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.argsctx.sim_paths_ctx_argr', reload=False);

logging.info('creating similar proposition context argument intersection and set minus matrices.');
# adjust dimensions, in case they are different
if mat_arg_l_ctx_sim_paths.shape[1] < mat_arg_r_ctx_sim_paths.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_l_ctx_sim_paths):
        mat_arg_l_ctx_sim_paths = mat_arg_l_ctx_sim_paths.todok();
    mat_arg_l_ctx_sim_paths.resize(mat_arg_r_ctx_sim_paths.shape);
if mat_arg_r_ctx_sim_paths.shape[1] < mat_arg_l_ctx_sim_paths.shape[1]:
    if sparse.isspmatrix_coo(mat_arg_r_ctx_sim_paths):
        mat_arg_r_ctx_sim_paths = mat_arg_r_ctx_sim_paths.todok();
    mat_arg_r_ctx_sim_paths.resize(mat_arg_l_ctx_sim_paths.shape);    

if not sparse.isspmatrix_coo(mat_arg_l_ctx_sim_paths):
    mat_arg_l_ctx_sim_paths = mat_arg_l_ctx_sim_paths.tocoo();
if not sparse.isspmatrix_coo(mat_arg_r_ctx_sim_paths):
    mat_arg_r_ctx_sim_paths = mat_arg_r_ctx_sim_paths.tocoo();
#
mat_arg_l_ctx_sim_paths = mat_arg_l_ctx_sim_paths.astype(bool);
mat_arg_r_ctx_sim_paths = mat_arg_r_ctx_sim_paths.astype(bool);
mat_arg_ctx_sim_union_arg = mat_arg_l_ctx_sim_paths + mat_arg_r_ctx_sim_paths;
mat_arg_ctx_sim_diff_arg = mat_arg_l_ctx_sim_paths != mat_arg_r_ctx_sim_paths;
mat_arg_ctx_sim_inters_arg = mat_arg_ctx_sim_union_arg - mat_arg_ctx_sim_diff_arg;
mat_arg_ctx_sim_l_minus_r_arg = mat_arg_ctx_sim_union_arg - mat_arg_r_ctx_sim_paths;
mat_arg_ctx_sim_r_minus_l_arg = mat_arg_ctx_sim_union_arg - mat_arg_l_ctx_sim_paths;

### ### ### ###
  ### ### ### ###
### ### ### ###

logging.info('stacking matrices.');
# stack only two matrices at a time because of memory issues
mat = sparse.csr_matrix((len(d_triples),1),dtype=np.float64); #  initialize with (num_rows x 1) matrix, a dummy vector because scipy.sparse does not allow to stack empty matrices

# feat. between arg_l and arg_r
#mat = sparse.hstack((mat, mat_paths));
#mat = sparse.hstack((mat, mat_sim_pairs)).tocsr();
## context feat. of arg_l and arg_r individually
#mat = sparse.hstack((mat, mat_arg_l_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_r_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_union_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_diff_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_inters_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_l_minus_r_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_r_minus_l_ctx.astype(np.float64))).tocsr();
## topic feat. of arg_l and arg_r
#mat = sparse.hstack((mat, mat_topic));
#mat = sparse.hstack((mat, mat_arg_l_topic));
#mat = sparse.hstack((mat, mat_arg_r_topic)).tocsr();
## similarity feat. of arg_l and arg_r individually
#mat = sparse.hstack((mat, mat_sim_arg_l.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_arg_r.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_union_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_diff_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_inters_arg .astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_l_minus_r_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_r_minus_l_arg.astype(np.float64))).tocsr();
# feat from example contexts
#mat = sparse.hstack((mat, mat_arg_l_ctx_paths.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_r_ctx_paths.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_union_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_diff_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_inters_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_l_minus_r_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_r_minus_l_arg.astype(np.float64))).tocsr();
# feat from similar context examples
#mat = sparse.hstack((mat, mat_arg_l_ctx_sim_paths.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_r_ctx_sim_paths.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_sim_union_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_sim_diff_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_sim_inters_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_sim_l_minus_r_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_ctx_sim_r_minus_l_arg.astype(np.float64))).tocsr();

mat = mat.tocsr()[:,1:]; # remove the first dummy vector

tc.run_classification_test(mat, true_labels, binarize=True, percentage_train=0.8, print_train_test_set_stat=True, test_thresholds=False, random_seed=623519, d_args=d_triples);