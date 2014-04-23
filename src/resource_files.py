# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:34:40 2014

@author: stevo
"""
###
server_args = '/home/steffen/data/nlkg/args_unique_v4.tsv';
server_args_w_ctx = '/home/steffen/data/nlkg/arg_full.tsv';
server_preds_w_ctx = '/home/steffen/data/nlkg/pred_full.tsv';

server_svo_flipped_counts = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.counts.relations_flipped.txt.gz'; # svo_invert.counts_lmi_pruned.relations_flipped_filtered.txt.gz;; -- too small
server_svo_flipped_dt = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.dt.relations_flipped_filtered.txt.gz';
server_svo_counts = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.counts_lmi_pruned.relations_filtered.txt.gz'; #svo_invert.counts.relations.txt.gz'; -- too big
server_svo_dt = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.dt.relations_filtered.txt.gz';
server_svo_lda_w2t = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.dt.relations.lda_5_5/model-final.doc2topic_';
server_svo_flipped_lda_w2t = '/home/steffen/data/nlkg/svo-triples/dt/svo_invert.dt.relations_flipped.lda_5_5/model-final.doc2topic_';

server_gquad_flipped_counts = '/home/steffen/data/nlkg/googlesyntactics/quadarcs.counts.relations_flipped.txt.gz';
server_gquad_flipped_dt = '/home/steffen/data/nlkg/googlesyntactics/quadarcs.dt.relations_flipped.txt.gz';
server_gquad_counts = '/home/steffen/data/nlkg/googlesyntactics/quadarcs.counts.relations.txt.gz';
server_gquad_dt = '/home/steffen/data/nlkg/googlesyntactics/quadarcs.dt.relations.txt.gz';

server_gfull_flipped_counts = '/home/steffen/data/nlkg/googlesyntactics/full.counts.relations_flipped.txt.gz';
server_gfull_flipped_dt = '/home/steffen/data/nlkg/googlesyntactics/full.dt.relations_flipped.txt.gz';
server_gfull_counts = '/home/steffen/data/nlkg/googlesyntactics/full.counts.relations.txt.gz';
server_gfull_dt = '/home/steffen/data/nlkg/googlesyntactics/full.dt.relations.txt.gz';

###
local_args = '../data/updates/6/args_unique_v4.tsv';
local_args_w_ctx = '../data/updates/4/arg_full.tsv';
local_preds_w_ctx = '../data/updates/4/pred_full.tsv';

local_nodes_flipped_counts = '../data/dt/nodes/nodes_invert.counts.relations_flipped.txt';
local_nodes_flipped_dt = '../data/dt/nodes/nodes_invert.dt.relations_flipped.filtered.txt';
local_nodes_counts = '../data/dt/nodes/nodes_invert.counts.relations.txt';
local_nodes_dt = '../data/dt/nodes/nodes_invert.dt.relations.filtered.txt.gz';

local_svo_flipped_counts = '../data/dt/svo/svo_invert.counts.relations_flipped.txt.gz';
local_svo_flipped_dt = '../data/dt/svo/svo_invert.dt.relations_flipped_filtered.txt.gz';
local_svo_counts = '../data/dt/svo/svo_invert.counts.relations_filtered.txt.gz';
local_svo_dt = '../data/dt/svo/svo_invert.dt.relations_filtered.txt.gz';
local_svo_lda_w2t = '../data/dt/svo/svo_invert.dt.relations.lda_5_5.model-final.doc2topic';
local_svo_flipped_lda_w2t = '../data/dt/svo/svo_invert.dt.relations_flipped.lda_5_5.model-final.doc2topic';

