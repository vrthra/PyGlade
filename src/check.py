import sys, imp
#import parse_ # does not have the .in_ construction for taints.
#parse_ = imp.new_module('parse_')

#def init_module(src):
#    with open(src) as sf:
#        exec(sf.read(), parse_.__dict__)

exec_map = {}
def check(s, label=None):
    if s in exec_map: return exec_map[s]
    v =  _check(s)
    #print("\t\t", repr(s), v, ' from: %s' % str(label))
    exec_map[s] = v
    return v

import re
def _check(s):
    try:
        match = re.match( r'[(]1(2|3)*4[)]', s)
        #parse_.main(s)
        if match: return True
        return False
    except:
        return False
