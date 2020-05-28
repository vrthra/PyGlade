import sys, imp

exec_map = {}
def check(s, label=None):
    if s in exec_map: return exec_map[s]
    v =  _check(s)
    #print("\t\t", repr(s), v, ' from: %s' % str(label))
    exec_map[s] = v
    return v

# This is the oracle. For now, we simply use a regex matcher
# but you can replace it with any context free oracle. Return
# True if your oracle agrees with the input.
import re
def _check(s):
    try:
        match = re.match( r'[(]1(2|3)*4[)]', s)
        #parse_.main(s)
        if match: return True
        return False
    except:
        return False
