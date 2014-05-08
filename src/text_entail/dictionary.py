# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 09:32:32 2014

@author: stevo
"""

class Dict():

	UNK = '<unk>';
	
	def __init__(self):
		self._id2w = [];
		self._w2id = {};    
	
	def add(self, w):
		if w in self._w2id:
			return self._w2id[w];
		id_ = len(self._w2id);
		self._w2id[w] = id_;
		self._id2w.append(w);
		return id_;
	
	def getid(self, w):
		if w in self._w2id:
			return self._w2id[w];
		return -1;

	def getword(self, i):
		if i < len(self._id2w) and i >= 0:
			return self._id2w[i];
		return Dict.UNK;
		
	def __len__(self):
		return len(self._id2w);
		
	def __contains__(self, w):
		return w in self._w2id;
		
class TupleDict():

	UNK = ('<unk>','<unk>');
	
	def __init__(self):
		self._id2t = [];
		self._t2id = {};
		self._l2id = {};
		self._r2id = {};
	
	def add(self, t):
		if t in self._t2id:
			return self._t2id[t];
		id_ = len(self._t2id);
		# add tuple
		self._t2id[t] = id_;
		self._id2t.append(t);
		# add left and right components
		l,r = t;
		if l not in self._l2id:
			self._l2id[l] = [];
		self._l2id[l].append(id_);
		if r not in self._r2id:
			self._r2id[r] = [];
		self._r2id[r].append(id_);
		return id_;
	
	def get_pair_id(self, t):
		if t in self._t2id:
			return self._t2id[t];
		return -1;

	def get_pair(self, i):
		if i < len(self._id2t) and i >= 0:
			return self._id2t[i];
		return TupleDict.UNK;
		
	def get_left_element_ids(self, leftelement):
		if leftelement in self._l2id:
			return self._l2id[leftelement];
		return [-1,];
		
	def get_right_element_ids(self, rightelement):
		if rightelement in self._r2id:
			return self._r2id[rightelement];
		return [-1,];

	def __len__(self):
		return len(self._id2t);
		
	def __contains__(self, x):
		return x in self._t2id or x in self._l2id or x in self._r2id;


class TripleDict():

	UNK = ('<unk>','<unk>','<unk>');
	
	def __init__(self):
		self._id2triple = [];
		self._triple2id = {};
		self._rtuple2ids = {};
		self._ltuple2ids = {};
		self._l2ids = {};
		self._m2ids = {};
		self._r2ids = {};
	
	def add(self, triple):
		if triple in self._triple2id:
			return self._triple2id[triple];
		id_ = len(self._id2triple);
		# add triple
		self._id2triple.append(triple);
		self._triple2id[triple] = id_;
		# add left and right tuple components
		l,m,r = triple;
		l_tuple = (l,m);
		r_tuple = (m,r);
		if l_tuple not in self._ltuple2ids:
			self._ltuple2ids[l_tuple] = [];
		self._ltuple2ids[l_tuple].append(id_);
		if r_tuple not in self._rtuple2ids:
			self._rtuple2ids[r_tuple] = [];
		self._rtuple2ids[r_tuple].append(id_);
		
		# add single left, middle, right components
		if l not in self._l2ids:
			self._l2ids[l] = [];
		self._l2ids[l].append(id_);
		if m not in self._m2ids:
			self._m2ids[m] = [];
		self._m2ids[m].append(id_);
		if r not in self._r2ids:
			self._r2ids[r] = [];
		self._r2ids[r].append(id_);
		
		return id_;
	
	def get_triple_id(self, triple):
		if triple in self._triple2id:
			return self._triple2id[triple];
		return -1;

	def get_triple(self, i):
		if i < len(self._id2triple) and i >= 0:
			return self._id2triple[i];
		return TripleDict.UNK;

	def get_left_tuple_ids(self, left_tuple):
		if left_tuple in self._ltuple2ids:
			return self._ltuple2ids[left_tuple];
		return [-1,];
		
	def get_right_tuple_ids(self, right_tuple):
		if right_tuple in self._rtuple2ids:
			return self._rtuple2ids[right_tuple];
		return [-1,];
		
	def get_left_element_ids(self, leftelement):
		if leftelement in self._l2ids:
			return self._l2ids[leftelement];
		return [-1,];

	def get_middle_element_ids(self, middle_element):
		if middle_element in self._m2ids:
			return self._m2ids[middle_element];
		return [-1,];
		
	def get_right_element_ids(self, rightelement):
		if rightelement in self._r2ids:
			return self._r2ids[rightelement];
		return [-1,];

	def __len__(self):
		return len(self._id2triple);
		
	def __contains__(self, x):
		return x in self._triple2id or x in self._ltuple2ids or x in self._rtuple2ids or x in self._l2ids or x in self._r2ids;