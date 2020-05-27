#!/usr/bin/env python
import json
import random
import sys
import itertools

import check

# What samples to use for a{n} to conirm that a* is a valid regex.
SAMPLES_FOR_REP = [0, 1, 2]

class Regex:
    def to_rules(self):
        if isinstance(self, Alt):
            for a1 in self.a1.to_rules():
                yield a1
            for a2 in self.a2.to_rules():
                yield a2
        elif  isinstance(self, Rep):
            for a3 in self.a.to_rules():
                for n in SAMPLES_FOR_REP:
                    yield a3 * n
        elif  isinstance(self, Seq):
            for a4 in self.arr[0].to_rules():
                if self.arr[1:]:
                    for a5 in Seq(self.arr[1:]).to_rules():
                        yield a4 + a5
                else:
                    yield a4

        elif  isinstance(self, One):
            assert not isinstance(self.o, Regex)
            yield self.o
        else:
            assert False

    def __str__(self):
        if isinstance(self, Alt):
            return "(%s|%s)" % (str(self.a1), str(self.a2))
        elif  isinstance(self, Rep):
            return "(%s)*" % self.a
        elif  isinstance(self, Seq):
            if len(self.arr) == 1:
                return "(%s)" % ''.join(str(a) for a in self.arr)
            else:
                return "(%s)" % ''.join(str(a) for a in self.arr)
        elif  isinstance(self, One):
            return ''.join(str(o).replace('*', '[*]').replace('(', '[(]').replace(')', '[)]') for o in self.o)
        else:
            assert False

class Alt(Regex):
    def __init__(self, a1, a2): self.a1, self.a2 = a1, a2
    def __repr__(self): return "(%s|%s)" % (self.a1, self.a2)
class Rep(Regex):
    def __init__(self, a): self.a = a
    def __repr__(self): return "(%s)*" % self.a
class Seq(Regex):
    def __init__(self, arr): self.arr = arr
    def __repr__(self): return "(%s)" % ' '.join([repr(a) for a in self.arr if a])
class One(Regex):
    def __init__(self, o): self.o = o
    def __repr__(self): return repr(self.o) if self.o else ''


# Alternations: If generalizing P alt[alpha]Q, then
# for  each decomposition alpha = a_1 a_2, where a_1 != [] and
# a_2 != [], generate P (rep[alpha_1] + alt[alpha_2]) Q
# ...
# in both cases, P alpha Q is also generated
# + is alternation i.e `|' in regular expression

# Ordering: If generalizing P alt[alpha] Q we prioritize shorter
# alpha_1.
# In either case, P alpha Q is ranked last
# Note that candidate repetitions and candidate alternations can
# be ordered independently
def gen_alt(alpha):
    length = len(alpha)
    # alpha_1 != e and alpha_2 != e
    for i in range(1,length): # shorter alpha_1 prioritized
        alpha_1, alpha_2 = alpha[:i], alpha[i:]
        assert alpha_1
        assert alpha_2
        for a1 in gen_rep(alpha_1):
            for a2 in gen_alt(alpha_2):
                yield Alt(a1, a2)
    if length: # this is the final choice.
        yield One(alpha)
    return


# Repetitions: If generalizing P rep[alpha]Q, then
# for  each decomposition alpha = a_1 a_2 a_3 such that
# a_2 != [], generate P alpha_1(alt[alpha_2])* rep[alpha_3] Q
# ...
# in both cases, P alpha Q is also generated

# Ordering: If generalizing P rep[alpha] Q we prioritize shorter
# alpha_1 since alpha_1 is not further generalized. Second, we
# prioritize longer alpha_2
# In either case, P alpha Q is ranked last
def gen_rep(alpha):
    length = len(alpha)
    for i in range(length): # shorter alpha1 prioritized
        alpha_1 = alpha[:i]
        # alpha_2 != e
        for j in range(i+1, length+1): # longer alpha2 prioritized
            alpha_2, alpha_3 = alpha[i:j], alpha[j:]
            assert alpha_2
            for a2 in gen_alt(alpha_2):
                for a3 in gen_rep(alpha_3):
                    yield Seq([One(alpha_1), Rep(a2), a3])
                if not alpha_3:
                    yield Seq([One(alpha_1), Rep(a2)])
    if length: # the final choice
        yield One(alpha)
    return

def to_strings(regex):
    """
    We are given the toekn, and the regex that is being checked to see if it
    is the correct abstraction. Hence, we first generate all possible rules
    that can result from this regex.
    The complication is that str_db contains multiple alternative strings for
    each token. Hence, we have to generate a combination of all these strings
    and try to check.
   """
    for rule in regex.to_rules():
        exp_lst_of_lsts = [list(str_db.get(token, [token])) for token in rule]
        for lst in exp_lst_of_lsts: assert lst
        for lst in itertools.product(*exp_lst_of_lsts):
            """
            We first obtain the expansion string by replacing all tokens with
            candidates, then reconstruct the string from the derivation tree by
            recursively traversing and replacing any node that corresponds to nt
            with the expanded string.
            """
            expansion = ''.join(lst)
            print("Expansion %s:\tregex:%s" % (repr(expansion), str(regex)))
            yield expansion

str_db = {}
regex_map = {}


def phase_1(alpha_in):
    # active learning of regular righthandside from bastani et al.
    # the idea is as follows: We choose a single nt to refine, and a single
    # alternative at a time.
    # Then, consider that single alternative as a sting, with each token a
    # character. Then apply regular expression synthesis to determine the
    # abstraction candiates. Place each abstraction candidate as the replacement
    # for that nt, and generate the minimum string. Evaluate and verify that
    # the string is accepted (adv: verify that the derivation tree is
    # as expected). Do this for each alternative, and we have the list of actual
    # alternatives.

    # seed input alpha_in is annotated rep(alpha_in)
    # Then, each generalization step selects a single bracketed substring
    # T[alpha] and generates candiates based on decompositions of alpha
    # i.e. an expression of alpha as alpha = a_1, a_2, ..a_k

    for regex in gen_rep(alpha_in):
        all_true = False
        for expr in to_strings(regex):
            if regex_map.get(regex, False):
                v = check.check(expr, regex)
                regex_map[regex] = v
                if not v: # this regex failed
                    #print('X', regex)
                    all_true = False
                    break # one sample of regex failed. Exit
            elif regex not in regex_map:
                v = check.check(expr, regex)
                regex_map[regex] = v
                if not v: # this regex failed.
                    #print('X', regex)
                    all_true = False
                    break # one sample of regex failed. Exit
            all_true = True
        if all_true: # get the first regex that covers all samples.
            #print("nt:", nt, 'rule:', str(regex))
            return regex
    #raise Exception() # this should never happen. At least one -- the original --  should succeed
    return None

    for k in regex_map:
        if regex_map[k]:
            print('->        ', str(k), file=sys.stderr)
    print('', file=sys.stderr)
    regex_map.clear()
    sys.stdout.flush()

def to_key(prefix):
    return 'k' + ''.join([str(s) for s in prefix])

# if step i generalizes P rep[alpha] Q to
# P alpha_1 (alt[alpha_2])* rep[alpha_3] Q
# we generate productions
# A_i -> alpha_1 A'_i A_k
# A'i -> \e + A'_i A_j
# equivalent to A_i -> alpha_1 A_j* A_k
# whre A_k comes from rep[alpha_3] and
# A_j comes from alt[Alpha_2]


# If step i generalizes P alt[alpha] Q to
# P (rep[alpha_1] + alt[alpha_2]) Q
# we include production
# A_i -> A_j + A_k
# where A_j comes from rep[alpha_1] and
# A_k comes from alt[alpha_2]

def extract_seq(regex, prefix):
    # Each item gets its own grammar with prefix.
    g = {}
    rule = []
    for i,item in enumerate(regex.arr):
        g_, k = extract_grammar(item, prefix + [i])
        g.update(g_)
        rule.append(k)
    g[to_key(prefix)] = [rule]
    return g, to_key(prefix)

def extract_rep(regex, prefix):
    # a
    g, k = extract_grammar(regex.a, prefix + [0])
    g[to_key(prefix)] = [[to_key(prefix), k], []]
    return g, to_key(prefix)

def extract_alt(regex, prefix):
    # a1, a2
    g1, k1 = extract_grammar(regex.a1, prefix + [0])
    g2, k2 = extract_grammar(regex.a2, prefix + [1])
    g = {**g1, **g2}
    g[to_key(prefix)] = [[k1], [k2]]
    return g, to_key(prefix)

def extract_one(regex, prefix):
    # one is not a non terminal
    return {}, ''.join(regex.o)

def phase_2(regex):
    # the basic idea is to first translate the regexp into a
    # CFG, where the terminal symbols are the symbols in the
    # regex, and the generalization steps are nonterminals
    # and next, to equate the nonterminals in that grammar
    # to each other
    # Alt, Rep, Seq, One
    prefix = [0]
    g, k = extract_grammar(regex, prefix)
    return g, k

def extract_grammar(regex, prefix):
    if isinstance(regex, Rep):
        return extract_rep(regex, prefix)
    elif isinstance(regex, Alt):
        return extract_alt(regex, prefix)
    elif isinstance(regex, Seq):
        return extract_seq(regex, prefix)
    elif isinstance(regex, One):
        return extract_one(regex, prefix)
    assert False

def main(inp):
    # phase 1
    regex = phase_1([i for i in inp])
    print(regex)
    cfg, start = phase_2(regex)
    print('Start: ', start)
    for k in cfg:
        print("%s ::= " % k)
        for alt in cfg[k]:
            print("   | " + ' '.join(alt))

if __name__ == '__main__':
    # we assume check is modified to include the
    # necessary oracle
    main(sys.argv[1])
