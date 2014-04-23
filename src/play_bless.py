# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:44:30 2014

@author: stevo
"""

from __future__ import print_function;
from __future__ import division;
import logging;
import text_entail.matrix as tm;
import text_entail.dictionary as td;
import text_entail.classify as tc;
import scipy.sparse as sparse;
import numpy as np;

#basedir =               '/home/steffen/data/bless/';
basedir =               '/Users/stevo/Workspaces/rumblejungle/n2n/data/bless/';

labels =                basedir + 'BLESS_nouns_hyper_vs_all.tsv';

svo_counts =            basedir + 'dt/svo_invert.counts_lmi_pruned.relations_filtered.txt.gz';
svo_flipped_counts =    basedir + 'dt/svo_invert.counts_lmi_pruned.relations_flipped_filtered.txt.gz';

svo_dt =                basedir + 'dt/svo_invert.dt.relations_filtered.txt.gz';
svo_flipped_dt =        basedir + 'dt/svo_invert.dt.relations_flipped_filtered.txt.gz';

svo_lda_w2t =           basedir + 'dt/svo_invert.dt.relations.lda_5_5/model-final.doc2topic_';
svo_flipped_lda_w2t =   basedir + 'dt/svo_invert.dt.relations_flipped.lda_5_5/model-final.doc2topic_';


reload(logging); logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

logging.info('loading true labels');
true_labels, d_triples = tm.arg_l_arg_r_pairs_vector(\
    labels, file_contains_context=False, has_header=False);
    
    
## relation pair features
logging.info('loading paths between argument pairs');
d_paths = td.Dict();
mat_paths = tm.arg_l_arg_r_asjo_matrix(d_triples._rtuple2ids, \
    svo_flipped_counts,\
    len(d_triples),
    col_indices = d_paths, \
    mmfile_presuffix='.paths', reload=False);

logging.info('loading similar argument pairs');
d_sim_pairs = td.Dict();
mat_sim_pairs = tm.arg_l_arg_r_asjo_matrix(d_triples._rtuple2ids, \
    svo_flipped_dt,\
    len(d_triples),
    col_indices = d_sim_pairs, \
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.simpairs', reload=False);

## context features
logging.info('loading argument context matrices');
d_ctx = td.Dict();
mat_arg_l_ctx = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_ctx,
    svo_counts,\
    len(d_triples),\
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.ctx_w1', reload=False);
 
mat_arg_r_ctx = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_ctx,
    svo_counts,\
    len(d_triples),\
    transform_w2sig=lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],\
    mmfile_presuffix='.ctx_arg_w1', reload=False);

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
mat_topic = tm.arg_l_arg_r_to_topic_matrix(d_triples._rtuple2ids,\
    svo_flipped_lda_w2t,\
    len(d_triples), \
    mmfile_presuffix='.bless.topic_pairs', reload=False);
    
mat_arg_l_topic = tm.arg_to_topic_matrix(d_triples._m2ids,\
    svo_lda_w2t,\
    len(d_triples),\
    mmfile_presuffix='.bless.topic_w1', reload=False);

mat_arg_r_topic = tm.arg_to_topic_matrix(d_triples._r2ids,\
    svo_lda_w2t,\
    len(d_triples),\
    mmfile_presuffix='.bless.topic_w2', reload=False);

# distributionally similar args for each arg
logging.info('loading similar arguments.')
d_arg = td.Dict();
mat_sim_arg_l = tm.arg_asjo_matrix(d_triples._m2ids,\
    d_arg,
    svo_dt,\
    len(d_triples),\
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.sim_w1', reload=False);

mat_sim_arg_r = tm.arg_asjo_matrix(d_triples._r2ids,\
    d_arg,
    svo_dt,\
    len(d_triples),\
    transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], \
    mmfile_presuffix='.sim_w2', reload=False);

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

logging.info('stacking matrices.');
# stack only two matrices at a time because of memory issues, initialize with (num_rows x 1) matrix which must be removed later again
mat = sparse.csr_matrix((len(d_triples),1),dtype=np.float64);

#### feat. between arg_l and arg_r
#mat = sparse.hstack((mat, mat_paths));
#mat = sparse.hstack((mat, mat_sim_pairs));
#### context feat. of arg_l and arg_r individually
mat = sparse.hstack((mat, mat_arg_l_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_r_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_union_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_diff_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_inters_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_l_minus_r_ctx.astype(np.float64)));
#mat = sparse.hstack((mat, mat_arg_r_minus_l_ctx.astype(np.float64)));
#### topic feat. of arg_l and arg_r
#mat = sparse.hstack((mat, mat_topic));
#mat = sparse.hstack((mat, mat_arg_l_topic));
#mat = sparse.hstack((mat, mat_arg_r_topic)).tocsr();
#### similarity feat. of arg_l and arg_r individually
#mat = sparse.hstack((mat, mat_sim_arg_l.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_arg_r.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_union_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_diff_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_inters_arg .astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_l_minus_r_arg.astype(np.float64)));
#mat = sparse.hstack((mat, mat_sim_r_minus_l_arg.astype(np.float64)));

random_seed = 623519;
#random_seed = 234123;

mat = mat.tocsr()[:,1:];
model = tc.run_classification_test(mat, true_labels, binarize=True, percentage_train=0.8, print_train_test_set_stat=True, test_thresholds=False, random_seed=random_seed);
#model = tc.run_classification_test(mat.tocsr()[:,1:], true_labels, binarize=True, percentage_train=0.8, print_train_test_set_stat=True, test_thresholds=False, random_seed=random_seed);
#names = d_ctx._id2w
#l = zip(names, model.coef_[0])
#ls = sorted(l, reverse=True, key= lambda x: x[1]);

#interesting_id = 10;
#names = d_ctx._id2w
#x = zip(names, np.squeeze(np.array(mat[interesting_id,:].todense())));
#x = sorted([(x,i) for (i, x) in enumerate(x) if x[1] > 0], key= lambda x: x[0][1], reverse=True);

