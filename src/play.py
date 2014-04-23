# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:44:30 2014

@author: stevo
"""

import text_entail.classify as tc;

mat = sparse.csr_matrix((len(d_triples),1),dtype=np.float64); #  initialize with (num_rows x 1) matrix, a dummy vector because scipy.sparse does not allow to stack empty matrices

# feat. between arg_l and arg_r
mat = sparse.hstack((mat, mat_paths));
mat = sparse.hstack((mat, mat_sim_pairs)).tocsr();
# context feat. of arg_l and arg_r individually
mat = sparse.hstack((mat, mat_arg_l_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_r_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_union_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_diff_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_inters_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_l_minus_r_ctx.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_r_minus_l_ctx.astype(np.float64))).tocsr();
# topic feat. of arg_l and arg_r
mat = sparse.hstack((mat, mat_topic));
mat = sparse.hstack((mat, mat_arg_l_topic));
mat = sparse.hstack((mat, mat_arg_r_topic)).tocsr();
# similarity feat. of arg_l and arg_r individually
mat = sparse.hstack((mat, mat_sim_arg_l.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_arg_r.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_union_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_diff_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_inters_arg .astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_l_minus_r_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_sim_r_minus_l_arg.astype(np.float64))).tocsr();
# feat from example contexts
mat = sparse.hstack((mat, mat_arg_l_ctx_paths.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_r_ctx_paths.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_union_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_diff_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_inters_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_l_minus_r_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_r_minus_l_arg.astype(np.float64))).tocsr();
# feat from similar context examples
mat = sparse.hstack((mat, mat_arg_l_ctx_sim_paths.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_r_ctx_sim_paths.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_sim_union_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_sim_diff_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_sim_inters_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_sim_l_minus_r_arg.astype(np.float64)));
mat = sparse.hstack((mat, mat_arg_ctx_sim_r_minus_l_arg.astype(np.float64))).tocsr();

mat = mat.tocsr()[:,1:]; # remove the first dummy vector

model = tc.run_classification_test(mat, true_labels, binarize=True, percentage_train=0.8, print_train_test_set_stat=True, test_thresholds=False, random_seed=623519, d_args=d_triples);

names = d_paths_ctx._id2w
l = zip(names, model.coef_[0])
ls = sorted(l, reverse=True, key= lambda x: x[1]);