# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 09:43:23 2014

@author: stevo
"""

from __future__ import print_function
from __future__ import division

import logging
import numpy as np
import random as rand

from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics

import text_entail.matrix as tm
import text_entail.io as tio

from IPython import embed

def run_baseline_classification_test( y_true ):
	"""
	"""
	# always true baseline
	print( "Baseline 'always true': " )
	y_pred = np.ones( len( y_true ))
	calculate_statistics( y_true, y_pred )

	# always false baseline
	print( "Baseline 'always false': " )
	y_true_ = np.ones( len( y_true )) - y_true
	calculate_statistics( y_true_, y_pred )

def run_classification_test( mat, y_true, binarize=True, 
	percentage_train=0.8, print_train_test_set_stat=True, 
	test_thresholds=False, random_seed=None, d_args=None ):
	"""
	"""
	# binarize full matrix if desired
	if binarize:
		logging.info( "binarizing feature matrix" )
		mat = tm.binarize_sparse_matrix( mat )
		logging.info( "finished binarizing feature matrix" )
	
	# create train and test split
	logging.info( "splitting data into train and test set" )
	idx_train, idx_test = get_stratified_train_test_indices( y_true, 
		percentage_train, random_seed )
	# idx_train, idx_test = get_train_test_indices_presplit( d_args );
	mat_train, mat_test, y_true_train, y_true_test = \
		split_matrix_to_train_and_test( mat, y_true, idx_train, idx_test, 
			print_stat=print_train_test_set_stat )

	model = classify( mat_train, mat_test, y_true_train, y_true_test, test_thresholds )
	return model

def get_stratified_train_test_indices( y_true, percentage_train=0.8, 
	random_seed=None ):
	"""
	"""
	r = rand.Random( x=random_seed )
	idx_pos = np.where( y_true > 0 )[0]
	r.shuffle( idx_pos )
	idx_neg = np.where( y_true < 1 )[0]
	r.shuffle( idx_neg )
	
	num_train_examples_pos = int( len( idx_pos ) * percentage_train )
	num_train_examples_neg = int( len( idx_neg ) * percentage_train )
	
	idx_train = np.hstack(( idx_pos[:num_train_examples_pos], 
							idx_neg[:num_train_examples_neg]))
	idx_test  = np.hstack(( idx_pos[num_train_examples_pos:], 
							idx_neg[num_train_examples_neg:]))
	
	# # re-shuffle to avoid having a separation of positive and negative samples
	# # note: unnecessary
	# r.shuffle( idx_train )
	# r.shuffle( idx_test )
	
	return idx_train, idx_test

def get_train_test_indices_presplit(d_args):

	t1 = tio.read_args_w_ctx('../data/updates/4/args_v2_am.tsv', has_header=False);
	train_ids = [];
	for ctx, arg_l, arg_r, __ in t1:
		id_ = d_args.get_triple_id((ctx, arg_l, arg_r));
		train_ids.append(id_);
	
	t2 = tio.read_args_w_ctx('../data/updates/4/args_v2_nz.tsv', has_header=False);
	test_ids = [];
	for ctx, arg_l, arg_r, __ in t2:
		id_ = d_args.get_triple_id((ctx, arg_l, arg_r));
		test_ids.append(id_);
	
	return train_ids, test_ids;
	
def split_matrix_to_train_and_test( mat, y_true, idx_train, idx_test, print_stat=False ):
	"""
	"""
	mat_train = mat[idx_train,:];
	labels_train = y_true[idx_train];
	mat_test = mat[idx_test,:];
	labels_test = y_true[idx_test];

	if print_stat:
		print('======');
		print('  percentage of training examples: {}\n  num training examples: {} ({}/{})\n  num testing examples:  {} ({}/{})'.format(
			len( idx_train ) / len( y_true ),
			len( idx_train ),
			len( np.where( labels_train > 0)[0] ),
			len( np.where( labels_train < 1)[0] ),
			len( idx_test ),
			len( np.where( labels_test > 0)[0] ),
			len( np.where( labels_test < 1)[0] )))
		print('======')
		
	return mat_train, mat_test, labels_train, labels_test

def classify( m_train, m_test, y_true_train, y_true_test, test_thresholds=False ):
	"""
	"""
	logging.info( "training classifier" )
	model = LogisticRegression( random_state=17, penalty='l1' )
	model.fit( m_train, y_true_train )
	logging.info( "finished training" )
	
	logging.info( "predicting" )
	y_pred = model.predict( m_test )
	logging.info( "finished predicting" )
	
	# stevo's custom statistics ( output mostly commented for now )
	calculate_statistics( y_true_test, y_pred )

	# default model score
	print( "\nmean accuracy %f\n" % model.score( m_test, y_pred ))

	# compute classification report ( incl. precision, recall, f_score etc. )
	report = metrics.classification_report( y_true_test, y_pred )
	print( report )

	# 
	if test_thresholds:

		logging.info( "testing thresholds" )
		y_pred = model.predict_proba( m_test )
		
		# Just take the probability of the positive classes
		y_pred = y_pred[:, 1]
		
		# From here on, it's just precision-recall calculations
		n = 10
		p_error = np.zeros( n+2 )
		n_error = np.zeros( n+2 )
		t = np.array([ -0.05 + float(i)/n for i in xrange(n+2) ])
		
		for i, e in enumerate( y_true_test ):
			prediction = y_pred[i]
			if e==1:
				p_error += ( np.sign( t - prediction ) + 1 ) / 2
				# p_error += ( -np.sign( t - prediction ) + 1 ) / 2
			else:
				n_error += (-np.sign( t - prediction ) + 1 ) / 2
				# n_error += ( np.sign( t - prediction ) + 1 ) / 2
		
		total_p = y_true_test.sum()
		total_n = y_true_test.shape[0] - total_p
		print( 'Number of positive examples:', total_p )
		print( 'Number of negative examples:', total_n )
		
		tp = total_p - p_error
		fn = p_error
		fp = n_error
		tn = total_n - n_error
		
		precision = tp / ( tp + fp )
		recall = tp / ( tp + fn )
		f1 = 2 * precision * recall / ( precision + recall )
		
		print( 'TP:', tp )
		print( 'FN:', fn )
		print( 'FP:', fp )
		print( 'TN:', tn )
		
		print( 'Precision:', precision )
		print( 'Recall:', recall )
		print( 'F1:', f1 )
		print( 'Best:', np.nanmax( f1 ))
		
	return model
	
def calculate_statistics( y_true, y_pred ):
	
	# embed()

	# calculate and print results
	y_true = y_true.astype( bool )
	y_pred = y_pred.astype( bool )

	# compute true positives, false positives, true negatives, false negatives
	# why would you apply bitwise operators on boolean values !?
	tp = y_true & y_pred
	fp = tp ^ y_pred
	fn = tp ^ y_true
	
	sum_tp = tp.sum();
	sum_fp = fp.sum();
	sum_fn = fn.sum();
	sum_tn = y_true.shape[0] - sum_tp - sum_fp - sum_fn;
	
	prec_1 = sum_tp / (sum_tp + sum_fp);
	rec_1  = sum_tp / (sum_tp + sum_fn);
	acc  = (sum_tp + sum_tn) / y_true.shape[0];
	f1_1   = 2 * (prec_1 * rec_1)/(prec_1 + rec_1);
	
	prec_0 = sum_tn / (sum_tn + sum_fn);
	rec_0  = sum_tn / (sum_tn + sum_fp);
	f1_0   = 2 * (prec_0 * rec_0)/(prec_0 + rec_0);
	
	prec = .5 * prec_1 + .5 * prec_0;
	rec = .5 * rec_1 + .5 * rec_0;
	f1   = 2 * (prec * rec)/(prec + rec);

	w1 = y_true.sum() / len(y_true);
	w2 = 1 - w1;
	prec_w = w1 * prec_1 + w2 * prec_0;
	rec_w = w1 * rec_1 + w2 * rec_0;
	f1_w   = 2 * (prec_w * rec_w)/(prec_w + rec_w);
	
	print('======');
	print('\n  TP: {}\n  FP: {}\n  FN: {}\n  TN: {}\n'.format(sum_tp, sum_fp, sum_fn, sum_tn));
	# print('  Acc: {}\n'.format(acc));
	# print('  Pr(1): {}\n  Re(1): {}\n  F1(1): {}\n'.format(prec_1, rec_1, f1_1));
	# print('  Pr(0): {}\n  Re(0): {}\n  F1(0): {}\n'.format(prec_0, rec_0, f1_0));
	# print('  Pr_w(0.5): {}\n  Re_w(0.5): {}\n  F1_w(0.5): {}\n'.format(prec, rec, f1));
	# print('  Pr_w(c): {}\n  Re_w(c): {}\n  F1_w(c): {}'.format(prec_w, rec_w, f1_w));
	print('======');

