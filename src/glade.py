#!/usr/bin/env python
import json
import random
import sys
import itertools
import copy
import check
import fuzz
import config

class Regex:
    def to_rules(self):
        if isinstance(self, Alt):
            if 9 == self.extra:
                for a1 in self.a1.to_rules():
                    yield a1
                for a2 in self.a2.to_rules():
                    yield a2
            else:  # It's part of the context, ignore rule.
                for a1 in Seq([self.a1, self.a2]).to_rules():
                    yield a1

        elif  isinstance(self, Rep):
            if 9 == self.extra:
                for a3 in self.a.to_rules():
                    for n in config.SAMPLES_FOR_REP:
                        yield a3 * n
            else:  # It's part of the context, ignore rule.
                for a3 in self.a.to_rules():
                    yield a3
        elif  isinstance(self, Seq):
            #print('Current arr: ', str(self.arr))
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
    def __init__(self, a1, a2, extra): 
        self.a1 = a1 
        self.a2 = a2
        self.extra = extra  # extra data used to mark if this object needs to be considered in the next check (if equals to 9) or not (equqls to 0). 
                            # That is, whether it's a part of the Context or not. See section 4.3
    def __repr__(self): return "(%s|%s)" % (self.a1, self.a2)
class Rep(Regex):
    def __init__(self, a, extra): 
        self.a = a
        self.extra = extra  # See section 4.3
    def __repr__(self): return "(%s)*" % self.a
class Seq(Regex):
    def __init__(self, arr): self.arr = arr
    def __repr__(self): return "(%s)" % ' '.join([repr(a) for a in self.arr if a])
class One(Regex):
    def __init__(self, o, extra): 
        self.o = o
        self.extra = extra # Substrings are annotated with extra data to express possible further generalization options.
                           # 0: for no generalization, 1: for Rep, 2: for Alt.
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
# We don't genralize all decendandts in one go, but only one substring at each step. Section 4.1 page 4
def gen_alt(alpha):
    length = len(alpha)
    # alpha_1 != e and alpha_2 != e
    for i in range(1,length): # shorter alpha_1 prioritized
        alpha_1, alpha_2 = alpha[:i], alpha[i:]
        assert alpha_1
        assert alpha_2
        yield Alt(One(alpha_1, 1), One(alpha_2, 2), 9)
    if length: # this is the final choice.
        yield One(alpha, 1)
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
# We don't genralize all decendandts in one go, but only one substring at each step. Section 4.1 page 4
def gen_rep(alpha):
    length = len(alpha)
    for i in range(length): # shorter alpha1 prioritized
        alpha_1 = alpha[:i]
        # alpha_2 != e
        for k in range(i+1, length+1): # longer alpha2 prioritized, see section 4.2
            j = length - (k - (i+1))   # j is the inverse of k.
            alpha_2, alpha_3 = alpha[i:j], alpha[j:]
            assert alpha_2
            yield Seq([One(alpha_1, 0), Rep(One(alpha_2, 2), 9), One(alpha_3, 1)])
    if length: # the final choice
        yield One(alpha, 0)
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
    print('Starting ... ')
    for rule in regex.to_rules():
        exp_lst_of_lsts = [list(str_db.get(token, [token])) for token in rule]
        #print('Current rule: ', str(rule))
        #print('Len of list: ', len(exp_lst_of_lsts))
        for lst in exp_lst_of_lsts: assert lst
        for lst in itertools.product(*exp_lst_of_lsts):
            """
            We first obtain the expansion string by replacing all tokens with
            candidates, then reconstruct the string from the derivation tree by
            recursively traversing and replacing any node that corresponds to nt
            with the expanded string.
            """
            expansion = ''.join(lst)
            #print("Expansion %s:\tregex:%s" % (repr(expansion), str(regex)))
            yield expansion

str_db = {}
regex_map = {}
regex_dict = dict()

def get_candidates(regex):
    regex1 = copy.deepcopy(regex)
    return traverse(regex1)


def get_checks(l_curr, candidate):
    return {}
def check_candidate(s):
    return True

# The traverse function is the generator of candidates, it's called at each step once, it selects a terminal substring 
# and generates all posssible generalization. Each representing a candidate regex.

def traverse(regex):
    exp = False # Used to insure that we don't modify more that one branch in each step.
    if isinstance(regex, Rep):
        print("It's a Rep")
        regex.extra = 0
        for x in traverse(regex.a):
            print("xxxxxxx")
            if x == -1: # -1 means we reached a leaf that is generizable.
                print("continue ...")
                continue
            else:
                yield Rep(x, 0)

    elif isinstance(regex, Alt):
        print("It's a Alt")
        regex.extra = 0
        for x in traverse(regex.a1):
            if x == -1:
                continue
            else:
                exp = True
                yield Alt(x, regex.a2, 0)
        if exp == False:
            for x in traverse(regex.a2):
                if x == -1:
                    continue
                else:
                    yield Alt(regex.a1, x, 0)

    elif isinstance(regex, Seq):
        print("It's a Seq")
        i = 0
        for obj in regex.arr:
            if exp == False:
                for x in traverse(obj):
                    if x == -1:
                        continue
                    else:
                        exp = True
                        ay = copy.deepcopy(regex.arr)
                        ay[i] = x
                        yield Seq(ay)
            i += 1

    elif isinstance(regex, One):
        print("It's a One")
        if regex.extra == 0:
            yield -1
        elif regex.extra == 1:
            for a in gen_rep(regex.o):
                yield a
            regex.extra = 0
        elif regex.extra == 2:
            for a in gen_alt(regex.o):
                yield a
            regex.extra = 0

# This helper function is here only to help print the regex heirarchy.
def get_dict(regex):
    suffix = str(random.randint(1, 999))
    if isinstance(regex, Rep):
        return {"Rep"+str(regex.extra): get_dict(regex.a)}
    elif isinstance(regex, Alt):
        return {"Alt"+str(regex.extra): [get_dict(regex.a1) ,get_dict(regex.a2)]}
    elif isinstance(regex, Seq):
        #return {"Seq": get_dict(regex.arr[0]), get_dict(regex.arr[0])} 
        return {"Seq": [get_dict(obj) for obj in regex.arr]}  	
    elif isinstance(regex, One):
        #return {"One": regex.o + str(regex.extra)}
        regex.o.insert(0, str(regex.extra))
        print(regex.o)
        return {"One": regex.o}       
    else:
        return "Nothing to return!"


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

    # Each iteration of the while loop corresponds to one generalization step.
    # Code below follows Algorithm 1, page 3 in the paper.

    done = False
    curr_reg = One(alpha_in, 1)

    while done == False:
        next_step = False
        started = False
        #regexw = copy.deepcopy(curr_reg)
        #print(get_dict(regexw))
        print(" ######## Next Step ########")
        # The traverse function supplies candidates, and is equivalent to the function "ConstructCandidates()" in the paper.
        for regex in traverse(curr_reg):
            started = True
            if regex == -1:
                print("---- Done ----")
                done = True
                break
            elif next_step == True:               
                break
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('Current Regex: ', str(regex))
            print('Current regex_map: ' + str(regex_map))
            all_true = False
            print('Number of exprs: ', len(list(to_strings(regex))))           
            print('Exprs: ', list(to_strings(regex)))
        
            # to_strings() function is equivlalent to the function ConstructChecks() in the paper.
            for expr in to_strings(regex):
                print('Current expression: ' + expr)
                
                if str(regex) in regex_map:
                    print('Regex tested already.')
                    all_true = False
                    break # Do not consider previous regexes as candidates. Exit
                elif str(regex) not in regex_map:
                    v = check.check(expr, regex)                    
                    if not v: # this regex failed.
                        #print('X', regex)
                        all_true = False
                        regex_map[str(regex)] = all_true
                        break # one sample of regex failed. Exit
                all_true = True
            if all_true: # get the first regex that covers all samples.
                #print("nt:", nt, 'rule:', str(regex))
                print("Accepted Regex.")
                regex_map[str(regex)] = all_true
                curr_reg = regex
                next_step = True

        if started == False:
            break
                
    #raise Exception() # this should never happen. At least one -- the original --  should succeed
    return curr_reg

    for k in regex_map:
        if regex_map[k]:
            print('->        ', str(k), file=sys.stderr)
    print('', file=sys.stderr)
    regex_map.clear()
    sys.stdout.flush()

def to_key(prefix, suffix=''):
    return '<k%s%s>'  % (''.join([str(s) for s in prefix]), suffix)

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
    g[to_key(prefix, '_rep')] = [[to_key(prefix, '_rep'), k], []]
    return g, to_key(prefix, '_rep')

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

def gen_new_grammar(a, b, key, cfg):
    # replace all instances of a and b with key
    new_g = {}
    for k in cfg:
        new_alts = []
        new_g[k] = new_alts
        for rule in cfg[k]:
            new_rule = [key if token in {a, b} else token for token in rule]
            new_alts.append(new_rule)
    rules = (new_g[a] + new_g[b])
    defs = {str(r):r for r in  rules}
    new_g[key] = [defs[l] for l in defs]
    return new_g

def consider_merging(a, b, key, cfg, start):
    g = gen_new_grammar(a, b, key, cfg)
    fzz = fuzz.LimitFuzzer(g)
    for i in range(config.P3Check):
        v = fzz.fuzz(start)
        r = check.check(v)
        if not r:
            return None
    return g

# the phase_3 is merging of keys
# The keys are unordered pairs of repetition keys A'_i, A'_j which corresponds
# to repetition subexpressions
def phase_3(cfg, start):
    # first collect all rep
    repetitions = [k for k in cfg if k.endswith('_rep>')]
    for i,(a,b) in enumerate(itertools.product(repetitions, repeat=2)):
        if a == b: continue
        c = to_key([i], '_')
        res = consider_merging(a,b, c, cfg, start)
        if res:
            print('Merged:', a, b, " => ", c)
            cfg = res
        else:
            continue
    return cfg

def main():
    # phase 1
    inputs = []
    regexes = []

    # We read inputs from a file.
    file1 = open('inputs', 'r') 
    Lines = file1.readlines() 

    for input in Lines:
        inputs.append(input.strip()) 

    if len(inputs) == 0:
        print("inputs file is empty! Please provide inputs.")
        sys.exit()
    for input in inputs:
        regexes.append(phase_1([i for i in input]))

    # Combine regexes into one regex as explained in Section 6.1
    regex = regexes[0]
    regexes.pop(0)
    for reg in regexes:
        regex = Alt(regex, reg, 0)

    print(regex)

    cfg, start = phase_2(regex)
    print('<start> ::= ', start)
    for k in cfg:
        print("%s ::= " % k)
        for alt in cfg[k]:
            print("   | " + ' '.join(alt))
    with open('grammar_.json', 'w+') as f:
        json.dump({'[start]': start, '[grammar]': cfg}, indent=4, fp=f)

    print('\nGrammar after Merging:\n')
    merged = phase_3(cfg, start)
    for k in merged:
        print("%s ::= " % k)
        for alt in merged[k]:
            print("   | " + ' '.join(alt))

    with open('grammar.json', 'w+') as f:
        json.dump({'[start]': start, '[grammar]': merged}, indent=4, fp=f)

if __name__ == '__main__':
    # we assume check is modified to include the
    # necessary oracle
    main()
