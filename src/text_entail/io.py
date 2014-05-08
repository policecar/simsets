# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 09:37:09 2014

@author: stevo
"""
from __future__ import print_function;
import os;
import logging;
#from text_entail.dictionary import Dict;
import gzip;

def read_args_wo_ctx(argfile, has_header=True):
	"""
	"""
	logging.info('reading file \'{}\''.format(argfile));
	
	with gzip.open(argfile) if argfile.endswith('.gz') else open(argfile) as f:
		if has_header:
			f.readline(); # skip header line
		for line in f:
			arg_l, arg_r, entail = line.rstrip().split('\t',2);
			yield arg_l, arg_r, entail;
	
	logging.info('finished reading file \'{}\''.format(argfile));

def read_args_w_ctx(argfile, has_header=True):
	"""
	"""
	logging.info('reading file \'{}\''.format(argfile));
	with gzip.open(argfile) if argfile.endswith('.gz') else open(argfile) as f:
		if has_header:
			f.readline(); # skip header line
		for line in f:
			ctx, arg_l, arg_r, entail = line.rstrip().split('\t',3);
			yield ctx, arg_l, arg_r, entail;
	logging.info('finished reading file \'{}\''.format(argfile));
	
def read_preds_w_ctx(predfile, has_header=True):
	"""
	"""
	# X	Y	PRED_L	   PRED_R	ENTAILING
	logging.info('reading file \'{}\''.format(predfile));
	with gzip.open(predfile) if predfile.endswith('.gz') else open(predfile) as f:
		if has_header:
			f.readline(); # skip header line
		for line in f:
			ctx_X, ctx_Y, pred_l, pred_r, entail = line.rstrip().split('\t',4);
			yield ctx_X, ctx_Y, pred_l, pred_r, entail;
	logging.info('finished reading file \'{}\''.format(predfile));

def read_jb_file(fn):
	"""
	"""
	logging.info('reading file \'{}\''.format(fn));
	with gzip.open(fn) if fn.endswith('.gz') else open(fn) as f:
		w_ = None;
		similar_w_ = None;
		for line in f:
			w, similar_w, sig = line.rstrip().split('\t',2);
			if '\t' in sig:
				sig = sig[:sig.find('\t')];
			if w != w_:
				if w_:
					yield w_, tuple(similar_w_);
				w_ = w;
				similar_w_ = [];
			similar_w_.append((similar_w, sig));
		if w_:
			yield w_, tuple(similar_w_);
	logging.info('finished reading file \'{}\''.format(fn));
			
#def read_dictrelevant_jb_file(fn, d):
#    with gzip.open(fn) if fn.endswith('.gz') else open(fn) as f:
#        w_ = None;
#        similar_w_ = None;
#        for line in f:
#            w, similar_w, sig = line.rstrip().split('\t',2);
#            if w not in d:
#                continue;
#            if '\t' in sig:
#                sig = sig[:sig.find('\t')];
#            if w != w_:
#                if w_:
#                    yield w_, tuple(similar_w_);
#                w_ = w;
#                similar_w_ = [];
#            similar_w_.append((similar_w, sig));
#        if w_:
#            yield w_, tuple(similar_w_);
			
def read_jb_file_filter_by_jo(fn, filter_by_jo_fun = lambda jo : False):
	"""
	"""
	logging.info('reading jb file \'{}\''.format(fn));
	with gzip.open(fn) if fn.endswith('.gz') else open(fn) as f:
		w_ = None;
		similar_w_ = None;
		for line in f:
			w, similar_w, sig = line.rstrip().split('\t',2);
			if not filter_by_jo_fun(w):
				continue;
			if '\t' in sig:
				sig = sig[:sig.find('\t')];
			if w != w_:
				if w_:
					yield w_, tuple(similar_w_);
				w_ = w;
				similar_w_ = [];
			similar_w_.append((similar_w, sig));
		if w_:
			yield w_, tuple(similar_w_);
	logging.info('finished reading jb file \'{}\''.format(fn));

#def read_args2dict(argfile, d=None):
#    logging.info('reading file \'{}\''.format(argfile));
#    if not d:
#        d = Dict();
#    with gzip.open(argfile) if argfile.endswith('.gz') else open(argfile) as f:
#        for line in f:
#            arg_l, arg_r, entail = line.rstrip().split('\t',2);
#            d.add(arg_l);
#            d.add(arg_r);
#    logging.info('finished file \'{}\''.format(argfile));
#    return d;
	
def binarize_mmfile(mm_file):
	"""
	"""
	logging.info('reading file \'{}\''.format(mm_file));
	new_mm_file = mm_file[:mm_file.rfind('.')] + '.bin' + mm_file[mm_file.rfind('.'):];
	logging.info('writing to file \'{}\''.format(new_mm_file));
	
	with open(mm_file) as f:
		with open(new_mm_file,'w') as fo:
			first_line_skipped = False;
			for line in f:
				if line.startswith('%'):
					print(line,file=fo,end='');
					continue;
				if not first_line_skipped:
					first_line_skipped = True;
					print(line,file=fo,end='');
					continue;
				i, j, v = line.split(' ',2);
				print('{} {} 1'.format(i, j),file=fo);
	logging.info('finished processing file \'{}\''.format(mm_file));
	logging.info('copying index file if existent \'{}\''.format(mm_file+'i'));
	if os.path.exists(mm_file+'i') and os.path.isfile(mm_file+'i'):
		os.popen('cp {} {}'.format(mm_file+'i', new_mm_file+'i'));

def read_word2topicfile(word2topic_file):
	"""
	"""
	logging.info('reading file \'{}\''.format(word2topic_file));
	with gzip.open(word2topic_file) if word2topic_file.endswith('.gz') else open(word2topic_file) as f:
		for line in f:
			word, topicid = line.rstrip().split('\t',1);
			yield word, int(topicid);
	logging.info('finished reading file \'{}\''.format(word2topic_file));
 
