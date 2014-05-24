#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:44:30 2014

@author: stevo
@author: priska
"""

from __future__ import print_function
from __future__ import division

from config import *
from pprint import pprint

import logging, os
import scipy.sparse as sparse
import numpy as np

import text_entail.matrix as tm
import text_entail.dictionary as td
import text_entail.classify as tc

try:
	import IPython
	from IPython import embed
except ImportError:
	pass

# filename specs for labels, contexts, similarities and topic modeling ( LDA )
# fn_labels   = os.path.join( BASE_DIR, '../bless/bless_nouns_coord_vs_rest.tsv' )
fn_labels   = os.path.join( BASE_DIR, '../bless/bless_nouns_hyper_vs_rest.tsv' )
# fn_labels   = os.path.join( BASE_DIR, '../bless/bless_nouns_mero_vs_rest.tsv' )

fn_ctx_word = os.path.join( BASE_DIR, 'ctx/ctx_lmi_pruned.gz' )
fn_ctx_pair = os.path.join( BASE_DIR, 'ctx/ctx_lmi_pruned_flipped.gz' )
fn_sim_word = os.path.join( BASE_DIR, 'sim/sim.gz' )
fn_sim_pair = os.path.join( BASE_DIR, 'sim/sim_flipped.gz' )
fn_lda_word = os.path.join( BASE_DIR, 'lda/model_final_doc2topic_5_5' )
fn_lda_pair = os.path.join( BASE_DIR, 'lda/model_final_doc2topic_5_5_flipped' )

refresh = True
ctx = True
sim = False
lda = False

# instantiate logger
reload( logging )
logging.basicConfig( format='%(asctime)s - %(message)s', level=logging.DEBUG )

logging.info( 'loading labels from file' )
y_true, d_triples = tm.arg_l_arg_r_pairs_vector( fn_labels, \
	file_contains_context=False, has_header=False )

num_triples = len( d_triples )

# Note: the prefix d_ indicates a dictionary, m_ a matrix, mb_ a boolean matrix

if ctx:

	logging.info( 'loading context features for word pairs' )
	d_ctx_pair = td.Dict()
	m_ctx_pair = tm.arg_l_arg_r_asjo_matrix( d_triples._rtuple2ids, fn_ctx_pair, 
		num_triples, col_indices=d_ctx_pair, mmfile_presuffix='_pairs', reload=refresh )

	logging.info( 'loading context features for words' )
	d_ctx_word = td.Dict()
	m_ctx_w1 = tm.arg_asjo_matrix( d_triples._m2ids, d_ctx_word, fn_ctx_word, num_triples,
		transform_w2sig=lambda w2sig: sorted( list( w2sig ), key = lambda x: float( x[1] ), reverse=True )[:20],
		mmfile_presuffix='_w1', reload=refresh )
	m_ctx_w2 = tm.arg_asjo_matrix( d_triples._r2ids, d_ctx_word, fn_ctx_word, num_triples, 
		transform_w2sig = lambda w2sig: sorted( list( w2sig ), key = lambda x: float( x[1] ), reverse=True )[:20], 
		mmfile_presuffix='_w2', reload=refresh )

	# adjust ( context ) matrix dimensions, if they vary
	if m_ctx_w1.shape[1] < m_ctx_w2.shape[1]:
		if sparse.isspmatrix_coo(m_ctx_w1):
			m_ctx_w1 = m_ctx_w1.todok()
		m_ctx_w1.resize(m_ctx_w2.shape)

	if m_ctx_w2.shape[1] < m_ctx_w1.shape[1]:
		if sparse.isspmatrix_coo(m_ctx_w2):
			m_ctx_w2 = m_ctx_w2.todok()
		m_ctx_w2.resize(m_ctx_w1.shape)

	if not sparse.isspmatrix_coo(m_ctx_w1):
		m_ctx_w1 = m_ctx_w1.tocoo()
	if not sparse.isspmatrix_coo(m_ctx_w2):
		m_ctx_w2 = m_ctx_w2.tocoo()

	logging.info( "computing set operations on context matrices " )
	mb_ctx_w1 				= m_ctx_w1.astype( bool )
	mb_ctx_w2 				= m_ctx_w2.astype( bool )
	mb_ctx_union_w1_w2		= mb_ctx_w1 + mb_ctx_w2
	mb_ctx_diff_w1_w2		= mb_ctx_w1 != mb_ctx_w2
	mb_ctx_intersect_w1_w2	= mb_ctx_union_w1_w2 - mb_ctx_diff_w1_w2
	mb_ctx_minus_w1_w2		= mb_ctx_union_w1_w2 - mb_ctx_w2
	mb_ctx_minus_w2_w1		= mb_ctx_union_w1_w2 - mb_ctx_w1

if lda:

	logging.info( 'loading topic features ( LDA ) for words and word pairs' )
	m_topic_pair = tm.arg_l_arg_r_to_topic_matrix( d_triples._rtuple2ids, fn_lda_pair, 
		num_triples, mmfile_presuffix='_pairs', reload=refresh )
	m_topic_w1 = tm.arg_to_topic_matrix( d_triples._m2ids, fn_lda_word, 
		num_triples, mmfile_presuffix='_w1', reload=refresh )
	m_topic_w2 = tm.arg_to_topic_matrix( d_triples._r2ids, fn_lda_word, 
		num_triples, mmfile_presuffix='_w2', reload=refresh )

if sim:

	logging.info( 'loading similarity features for word pairs' )
	d_sim_pair = td.Dict()
	m_sim_pair = tm.arg_l_arg_r_asjo_matrix( d_triples._rtuple2ids, fn_sim_pair, 
		num_triples, col_indices = d_sim_pair, 
		transform_w2sig=lambda w2sig: sorted( list(w2sig), key=lambda x: float( x[1] ), reverse=True )[:20],
		mmfile_presuffix='_pairs', reload=refresh )

	logging.info( 'loading similarity features for words' )
	d_sim_word = td.Dict()
	m_sim_w1 = tm.arg_asjo_matrix(d_triples._m2ids, d_sim_word, fn_sim_word, num_triples,
		transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20], 
		mmfile_presuffix='_w1', reload=refresh )
	m_sim_w2 = tm.arg_asjo_matrix(d_triples._r2ids, d_sim_word, fn_sim_word, num_triples, 
		transform_w2sig = lambda w2sig: sorted(list(w2sig), key=lambda x: float(x[1]), reverse=True)[:20],
		mmfile_presuffix='_w2', reload=refresh )

	# adjust ( similarity ) matrix dimensions, if they vary
	if m_sim_w1.shape[1] < m_sim_w2.shape[1]:
		if sparse.isspmatrix_coo(m_sim_w1):
			m_sim_w1 = m_sim_w1.todok()
		m_sim_w1.resize(m_sim_w2.shape)
	if m_sim_w2.shape[1] < m_sim_w1.shape[1]:
		if sparse.isspmatrix_coo(m_sim_w2):
			m_sim_w2 = m_sim_w2.todok()
		m_sim_w2.resize(m_sim_w1.shape)    

	if not sparse.isspmatrix_coo(m_sim_w1):
		m_sim_w1 = m_sim_w1.tocoo()
	if not sparse.isspmatrix_coo(m_sim_w2):
		m_sim_w2 = m_sim_w2.tocoo()
	
	logging.info( "computing set operations on similarity matrices" )
	mb_sim_w1 				= m_sim_w1.astype( bool )
	mb_sim_w2 				= m_sim_w2.astype( bool )
	mb_sim_union_w1_w2 		= mb_sim_w1 + mb_sim_w2
	mb_sim_diff_w1_w2 		= mb_sim_w1 != mb_sim_w2
	mb_sim_intersect_w1_w2 	= mb_sim_union_w1_w2 - mb_sim_diff_w1_w2
	mb_sim_minus_w1_w2 		= mb_sim_union_w1_w2 - mb_sim_w2
	mb_sim_minus_w2_w1 		= mb_sim_union_w1_w2 - mb_sim_w1

logging.info( 'stacking matrices' )
# stack only two matrices at a time because of memory issues, 
# initialize with ( num_rows x 1 ) matrix which must be removed later again
#TEUXDEUX: where is this removed again ?
mat = sparse.csr_matrix(( num_triples, 1 ), dtype=np.float64 )

#TEUXDEUX: does it make sense to stack boolean and non-boolean matrices !?
# context and similarity features for word pairs
mat = sparse.hstack(( mat, m_ctx_pair ))
# mat = sparse.hstack(( mat, m_sim_pair ))

embed()

# context features of words
mat = sparse.hstack(( mat, mb_ctx_w1.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_union_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_diff_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_intersect_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_minus_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_ctx_minus_w2_w1.astype( np.float64 )))

# topic features for words and for word pairs
# mat = sparse.hstack(( mat, m_topic_pair ))
# mat = sparse.hstack(( mat, m_topic_w1 ))
# mat = sparse.hstack(( mat, m_topic_w2 )).tocsr() # why here tocsr() !?

# similarity features for words
# mat = sparse.hstack(( mat, mb_sim_w1.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_union_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_diff_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_intersect_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_minus_w1_w2.astype( np.float64 )))
# mat = sparse.hstack(( mat, mb_sim_minus_w2_w1.astype( np.float64 )))

# pairs <set operation> single noun

rand_seed = 623519
# rand_seed = 234123

logging.info( "training classifier and predicting labels" )
mat = mat.tocsr()[:,1:]
model = tc.run_classification_test( mat, y_true, binarize=True, # True
	percentage_train=0.8, print_train_test_set_stat=True, 
	test_thresholds=False, random_seed=rand_seed, d_triples=d_triples )

# names = d_ctx_word._id2w
# l = zip( names, model.coef_[0] )
# ls = sorted( l, reverse=True, key= lambda x: x[1] )

# interesting_id = 10
# names = d_ctx_word._id2w
# x = zip( names, np.squeeze( np.array( mat[interesting_id,:].todense() )))
# x = sorted([ (x,i) for (i, x) in enumerate(x) if x[1] > 0 ], key= lambda x: x[0][1], reverse=True )

logging.info( "inspect some of the features" )
# replicate hstacking here to attain names
# names = d_ctx_word._id2w
names = d_ctx_pair._id2w + d_ctx_word._id2w
# names = d_ctx_pair._id2w + d_sim_pair._id2w
# names = d_topic_pair._id2w
# names = d_ctx_pair._id2w + d_ctx_word._id2w
# combine feature names with their coefficients /weights
features = zip( names, model.coef_[0] )
# sort descending
sorted_features = sorted( features, reverse=True, key=lambda x: x[1] )
# print top x features
pprint( sorted_features[:35] )

# embed()
