# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:34:20 2014

@author: stevo
"""

def get_proposition_as_triple(proposition):
    return get_lhs_arg(proposition), get_pred(proposition), get_rhs_arg(proposition);

def get_lhs_arg(proposition):
    return proposition[:proposition.find(' ')];

def get_rhs_arg(proposition):
    return proposition[proposition.rfind(' ')+1:];
    
def get_pred(proposition):
    return proposition[proposition.find(' ')+1:proposition.rfind(' ')];