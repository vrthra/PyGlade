#!/usr/bin/env python
import copy
import itertools
import json
import random
import sys

import check
import config
import fuzz

CHECKS = 0
UNMERGED_GRAMMAR = {}


class Regex:
    def to_rules(self):
        if isinstance(self, Alt):
            if self.newly_generalized:
                yield from self.a1.to_rules()
                yield from self.a2.to_rules()
                self.newly_generalized = False
            elif self.a1_gen or newly_generalized_descendant(self.a1):  # Expand the first alternative if it, or one of it's descendants were newly generalized.  
                yield from self.a1.to_rules()
            elif self.a2_gen or newly_generalized_descendant(self.a2):  # Expand the second alternative if it, or one of it's descendants were newly generalized. 
                yield from self.a2.to_rules()

            else:  # Else it's part of the context. We don't enumerate all rules, but pick randomely only one of the two component.
                x = random.choice([True, False])
                if x:
                    yield from self.a1.to_rules()
                else:
                    yield from self.a2.to_rules()

        elif isinstance(self, Rep):
            if self.newly_generalized:
                for a3 in self.a.to_rules():
                    for n in config.SAMPLES_FOR_REP:
                        yield a3 * n
                self.newly_generalized = False
            else:  # It's part of the context, ignore rule.
                for a3 in self.a.to_rules():
                    yield a3

        elif isinstance(self, Seq):
            for a4 in self.arr[0].to_rules():
                if self.arr[1:]:
                    for a5 in Seq(self.arr[1:]).to_rules():
                        yield a4 + a5
                else:
                    yield a4

        elif isinstance(self, One):
            assert not isinstance(self.o, Regex)
            yield self.o[-1] # return last added character, or the original character if none were added.
        else:
            assert False

    def __str__(self):
        if isinstance(self, Alt):
            return "(%s|%s)" % (str(self.a1), str(self.a2))
        elif isinstance(self, Rep):
            return "(%s)*" % self.a
        elif isinstance(self, Seq):
            if len(self.arr) == 1:
                return "(%s)" % ''.join(str(a) for a in self.arr)
            else:
                return "(%s)" % ''.join(str(a) for a in self.arr)
        elif isinstance(self, One):
            if len(self.o) > 1:
                return "(%s)" % '|'.join(str(o).replace('*', '[*]').replace('(', '[(]').replace(')', '[)]') for o in self.o)
            else:
                return ''.join(str(o).replace('*', '[*]').replace('(', '[(]').replace(')', '[)]') for o in self.o)
        elif isinstance(self, Alts):
            return "(%s)" % ' | '.join(str(a) for a in self.arr)
        else:
            assert False


class Alt(Regex):
    def __init__(self, a1, a2, extra, a1_gen=False, a2_gen=False):
        self.a1 = a1
        self.a2 = a2
        self.newly_generalized = extra  # extra data used to mark if this object needs to be considered in the next check (if True) or not (if False).
                                        # That is, whether it's a part of the Context or not. See section 4.3:
                                        # Residual capturing the portion of L tilde that is generalized compared to L hat.
        self.a1_gen = a1_gen
        self.a2_gen = a2_gen

    def __repr__(self):
        return "(%s|%s)" % (self.a1, self.a2)


class Rep(Regex):
    def __init__(self, a, extra):
        self.a = a
        self.newly_generalized = extra  # See section 4.3

    def __repr__(self):
        return "(%s)*" % self.a


class Seq(Regex):
    def __init__(self, arr):
        self.arr = arr

    def __repr__(self):
        return "(%s)" % ' '.join([repr(a) for a in self.arr if a])


class Alts(Regex):
    def __init__(self, arr):
        self.arr = arr

    def __repr__(self):
        return "(%s)" % ' | '.join([repr(a) for a in self.arr if a])


class One(Regex):
    def __init__(self, o, extra, generalized=0, curr_char_gen=False):
        self.o = o # A list containing the original character and all possible character replacements.
        self.next_gen = extra # Substrings are annotated with extra data to express possible further generalization options.
                              # 0: for no generalization, 1: for Rep, 2: for Alt.
                              # Section 4.1: These annotations indicate that the
                              # bracketed substring in the current regular expression can be
                              # generalized by adding either a repetition (if tau = rep) or an
                              # alternation (if tau = alt).
        self.generalized = generalized   # 127 if all possible character replacements have been tried.
        self.curr_char_gen = curr_char_gen # True if it's the currect treminal being generalized in the Char Generalization Phase.

    def __repr__(self):
        return "(%s)" % ' '.join([repr(a) for a in self.o if a])


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
# We don't genralize all descendants in one go, but only one substring at each step. Section 4.1 page 4
# each generalization step selects a single bracketed substring [\alpha]\tau and generates candidates based on decompositions of \alpha


def gen_alt(alpha):
    length = len(alpha)
    # alpha_1 != e and alpha_2 != e
    for i in range(1, length):  # shorter alpha_1 prioritized
        alpha_1, alpha_2 = alpha[:i], alpha[i:]
        assert alpha_1
        assert alpha_2
        yield Alt(One([alpha_1], 1), One([alpha_2], 2), True)
    if length:  # this is the final choice.
        yield One([alpha], 1)
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
    if length < 2:  # if alpha is a single char, then return it as is, see Figure 2, Step R8
        yield One([alpha], 0)
    else:
        for i in range(length):  # shorter alpha1 prioritized
            alpha_1 = alpha[:i]
            # alpha_2 != e
            for k in range(i + 1, length + 1):  # longer alpha2 prioritized, see section 4.2
                j = length - (k - (i + 1))      # j is the inverse of k.
                alpha_2, alpha_3 = alpha[i:j], alpha[j:]
                assert alpha_2
                if i == 0 and j == length:
                    yield Rep(One([alpha_2], 2), True)
                elif i != 0 and j != length:
                    yield Seq([One([alpha_1], 0), Rep(One([alpha_2], 2), True), One([alpha_3], 1)])
                elif i == 0 and j != length:
                    yield Seq([Rep(One([alpha_2], 2), True), One([alpha_3], 1)])
                elif i != 0 and j == length:
                    yield Seq([One([alpha_1], 0), Rep(One([alpha_2], 2), True)])
        if length:  # the final choice
            yield One([alpha], 0)
    return


# List of all printable ASCII characters.
all_chars = [chr(i) for i in range(128)]


def gen_char(regex):
    # This function traverses a regex, then finds a generalizable
    # unit (One object). Then adds one alternative char to it and return.
    if isinstance(regex, Rep):
        x = gen_char(regex.a)
        if x == NON_GENERALIZABLE:  # We reached a node that is non generalizable.
            return NON_GENERALIZABLE
        else:
            return Rep(x, False)

    elif isinstance(regex, Alt):
        x = gen_char(regex.a1)
        if x != NON_GENERALIZABLE:
            new_reg = Alt(x, regex.a2, False)
            new_reg.a1_gen = True
            return new_reg
        else:
            regex.a1_gen = False
            x = gen_char(regex.a2)
            if x != NON_GENERALIZABLE:
                new_reg = Alt(regex.a1, x, False)
                new_reg.a2_gen = True
                return new_reg
            else:
                regex.a1_gen = False
                regex.a2_gen = False
                return NON_GENERALIZABLE

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            x = gen_char(obj)
            if x != NON_GENERALIZABLE:
                ay = copy.deepcopy(regex.arr)
                ay[i] = x
                return Seq(ay)
            i += 1
        return NON_GENERALIZABLE

    elif isinstance(regex, One):
        global ROLL_BACK
        if ROLL_BACK and regex.curr_char_gen:
            # We remove the last added char from the list of alternatives.
            del regex.o[-1]
            ROLL_BACK = False
            return regex
        if regex.generalized > 126:
            # All chars have been tried, we mark current unit as non-generalizable.
            regex.curr_char_gen = False
            return NON_GENERALIZABLE
        else:
            # Here we perform a generalization step.
            regex.curr_char_gen = True
            curr_char = all_chars[regex.generalized]
            if curr_char != regex.o[0]:
                regex.o.append(curr_char)
            elif regex.generalized < 127:  # Try the next char in the list
                regex.generalized += 1
                curr_char = all_chars[regex.generalized]
                regex.o.append(curr_char)
            elif regex.generalized == 127:  # All chars have been tried
                regex.curr_char_gen = False
                return NON_GENERALIZABLE
            regex.generalized += 1
            return regex


def atomize(regex):
    # Before executing the Char Generalization Phase, we break
    # strings in regex into separate chars, that is,
    # given a One regex containing a string, we break it into
    # a Seq regex that contains multiple One regexes,
    # each containing a single char. This way we can systematically
    # generalize each char/teminal/sigma_i separately.

    if isinstance(regex, Rep):
        regex.a = atomize(regex.a)
        return regex

    elif isinstance(regex, Alt):
        regex.a1 = atomize(regex.a1)
        regex.a2 = atomize(regex.a2)
        return regex

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            obj = atomize(obj)
            regex.arr[i] = obj
            i += 1
        return regex

    elif isinstance(regex, One):
        regex_orig = copy.deepcopy(regex)
        stg = regex.o[0]
        if len(stg) > 1:
            regex.o.pop()
            arr = []
            for i in stg:
                arr.append(One([i], 0))
            return Seq(arr)
        return One(regex_orig.o[0], 0)


def newly_generalized_descendant(regex):
    # Check if one of regex children nodes or descendants has been newly generalized
    # The function is useful when constructing checks.

    if isinstance(regex, Rep):
        if regex.newly_generalized:
            return True
        else:
            return newly_generalized_descendant(regex.a)

    elif isinstance(regex, Alt):
        if regex.newly_generalized:
            return True
        elif newly_generalized_descendant(regex.a1) or newly_generalized_descendant(regex.a2):
            return True
        else:
            return False

    elif isinstance(regex, Seq):
        for obj in regex.arr:
            if newly_generalized_descendant(obj):
                return True
        return False

    elif isinstance(regex, One):
        return False


def del_double_rep(regex):
    # To make regex more compact and reduce depth: Given nested Alt objects.
    # we transform it into a single Alts object.

    if isinstance(regex, Rep):
        if isinstance(regex.a, Rep):
            regex = regex.a
            return del_double_rep(regex)
        else:
            child = del_double_rep(regex.a)
            return Rep(child, regex.newly_generalized)

    elif isinstance(regex, Alt):
        regex.a1 = del_double_rep(regex.a1)
        regex.a2 = del_double_rep(regex.a2)
        return regex

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            obj = del_double_rep(obj)
            regex.arr[i] = obj
            i += 1
        return regex

    elif isinstance(regex, One):
        return regex


def compact(regex):
    # To make regex more compact and reduce depth: Given nested Alt objects.
    # we transform it into a single Alts object.

    if isinstance(regex, Rep):
        regex.a = compact(regex.a)
        return regex

    elif isinstance(regex, Alt):
        e1 = compact(regex.a1)
        e2 = compact(regex.a2)
        if not isinstance(e1, Alts) and not isinstance(e2, Alts):
            return Alts([e1, e2])
        elif isinstance(e1, Alts) and not isinstance(e2, Alts):
            return Alts(e1.arr + [e2])
        elif not isinstance(e1, Alts) and isinstance(e2, Alts):
            return Alts([e1] + e2.arr)
        else:
            return Alts(e2.arr + e2.arr)

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            obj = compact(obj)
            regex.arr[i] = obj
            i += 1
        return regex

    elif isinstance(regex, One):
        return regex


ROLL_BACK = False  # Roll back last character generalization step.


def char_gen_phase(regex):
    # character generalization phase that generalizes
    # terminals in the synthesized regular expression R.
    # The algorithm considers generalizing each terminal
    # in the regex to every (different) terminal in Sigma.
    # Section 6.2 Page 8.
    global ROLL_BACK
    global CHECKS
    while True:
        # At each iteration, we first save the current regex before working on the regex.
        regexcp = copy.deepcopy(regex)
        regex = gen_char(regex)
        if regex == NON_GENERALIZABLE:
            # No more char generalizations are possible. We are done with Char Generalization Phase.
            # Return last successfully generalized regex.
            return regexcp
        else:
            exprs = list(to_strings(regex))
            for expr in exprs:
                CHECKS += 1
                v = check.check(expr, regex)
                if not v:  # this regex failed.
                    ROLL_BACK = True
                    gen_char(regex)
                    break  # one sample of regex failed. Exit
                else:
                    ROLL_BACK = False


def to_strings(regex):
    """
    We are given the token, and the regex that is being checked to see if it
    is the correct abstraction. Hence, we first generate all possible rules
    that can result from this regex.
    The complication is that str_db contains multiple alternative strings for
    each token. Hence, we have to generate a combination of all these strings
    and try to check.
    """
    for rule in regex.to_rules():
        exp_lst_of_lsts = [list(str_db.get(token, [token])) for token in rule]
        for lst in exp_lst_of_lsts:
            assert lst
        for lst in itertools.product(*exp_lst_of_lsts):
            """
            We first obtain the expansion string by replacing all tokens with
            candidates, then reconstruct the string from the derivation tree by
            recursively traversing and replacing any node that corresponds to nt
            with the expanded string.
            """
            expansion = ''.join(lst)
            yield expansion


str_db = {}
regex_map = {}
valid_regexes = set()
regex_dict = dict()
NON_GENERALIZABLE = -1


# The get_candidates function is the generator of candidates. It's called at each step once, it selects a terminal substring
# then generates all posssible generalization for that substring. Each representing a candidate regex.
def get_candidates(regex):
    exp = False  # Used to insure that we don't modify more that one branch in each step.
    if isinstance(regex, Rep):
        for x in get_candidates(regex.a):
            if x == NON_GENERALIZABLE:  # We reached a leaf that is non generalizable.
                continue
            else:
                yield Rep(x, False)

    elif isinstance(regex, Alt):
        for x in get_candidates(regex.a1):
            if x == NON_GENERALIZABLE:
                continue
            else:
                exp = True
                yield Alt(x, regex.a2, False)
        if not exp:
            for x in get_candidates(regex.a2):
                if x == NON_GENERALIZABLE:
                    continue
                else:
                    yield Alt(regex.a1, x, False)

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            if not exp:
                for x in get_candidates(obj):
                    if x == NON_GENERALIZABLE:
                        continue
                    else:
                        exp = True
                        ay = copy.deepcopy(regex.arr)
                        ay[i] = x
                        yield Seq(ay)
            i += 1

    elif isinstance(regex, One):
        if regex.next_gen == 0:
            yield NON_GENERALIZABLE
        elif regex.next_gen == 1:
            yield from gen_rep(regex.o[-1])
            regex.next_gen = 0
        elif regex.next_gen == 2:
            yield from gen_alt(regex.o[-1])
            regex.next_gen = 0


# This helper function is here only to help print the regex heirarchy.
def get_dict(regex):
    if isinstance(regex, Rep):
        return {"Rep": [get_dict(regex.a), regex.newly_generalized]}
    elif isinstance(regex, Alt):
        return {"Alt": [get_dict(regex.a1) , get_dict(regex.a2), regex.newly_generalized]}
    elif isinstance(regex, Seq):
        return {"Seq": [get_dict(obj) for obj in regex.arr]}
    elif isinstance(regex, One):
        regex.o.insert(0, str(regex.next_gen))
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
    curr_reg = One([alpha_in], 1)
    global CHECKS
    while not done:
        next_step = False
        started = False
        # The get_candidates function supplies candidates, and is equivalent to the function "ConstructCandidates()" in the paper.
        for regex in get_candidates(curr_reg):
            started = True
            if regex == NON_GENERALIZABLE:
                # No more generalizations are possible. We are done with Phase 1.
                done = True
                break
            elif next_step:
                # We go to the next generalization step.
                break
            all_true = False
            regex = del_double_rep(regex)
            # to_strings() function is equivlalent to the function ConstructChecks() in the paper.
            exprs = list(to_strings(regex))

            ay = copy.deepcopy(regex)
            ay = del_double_rep(ay)
            var = str(get_dict(ay))
            if var in valid_regexes:
                continue
            for expr in exprs:
                if str(regex) in regex_map:
                    all_true = regex_map[str(regex)]
                    break  # Do not consider previous regexes as candidates. Exit
                elif str(regex) not in regex_map:
                    CHECKS += 1
                    v = check.check(expr, regex)
                    if not v:  # this regex failed.
                        all_true = False
                        regex_map[str(regex)] = all_true
                        break  # one sample of regex failed. Exit
                all_true = True
            if all_true:  # get the first regex that covers all samples.
                regex_map[str(regex)] = all_true
                ayy = copy.deepcopy(regex)
                var = str(get_dict(ayy))
                valid_regexes.add(var)
                curr_reg = regex
                next_step = True

        if not started:
            break

    atomized_reg = atomize(curr_reg)
    final_reg = char_gen_phase(atomized_reg)
    compact_reg = compact(final_reg)
    return compact_reg


def to_key(prefix, suffix=''):
    return '<k%s%s>' % (''.join([str(s) for s in prefix]), suffix)


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
    for i, item in enumerate(regex.arr):
        g_, k = extract_grammar(item, prefix + [i])
        g.update(g_)
        rule.append(k)
    g[to_key(prefix)] = [rule]
    return g, to_key(prefix)


def extract_alts(regex, prefix):
    # a1, a2
    g = {}
    rules = []
    for i, item in enumerate(regex.arr):
        g_, k = extract_grammar(item, prefix + [i])
        g.update(g_)
        rules.append([k])

    g[to_key(prefix)] = rules
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
    if len(regex.o) == 1:  # one is not a non terminal
        return {}, ''.join(regex.o[0])
    else:  # Regex One is a non terminal, meaning it has been generalized to a list of n chars. Therefore we treat it as an Alt object with n alternatives. See example in section 6.2
        alts = []
        g = {}
        for t in regex.o:
            alts.append([t])
        g[to_key(prefix)] = alts
        return g, to_key(prefix)


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
    elif isinstance(regex, Alts):
        return extract_alts(regex, prefix)
    assert False


def change_nonterminal(a, b, cfg):
    # Function to overwrite the first rule of rep a, with the first rule of rep b. This needs to be done as part of the Check construction of Merging phase.
    new_g = copy.deepcopy(cfg)

    new_g[a][0][1] = new_g[b][0][1]
    return new_g


def gen_new_grammar(a, b, key, cfgx):
    included = False  # True if a or b was previously merged with other non-terminal.
    test = True  # False if (a,b) have already been merged indirectly, via transitivity.

    cfg = copy.deepcopy(cfgx)
    for k in cfg:
        if k.endswith('_>'):
            for rule in cfg[k]:
                if rule[0] == a:
                    if [b] not in cfg[k]:
                        cfg[k].append([b])
                    else:
                        test = False
                    included = True
                    break
                elif rule[0] == b:
                    if [a] not in cfg[k]:
                        cfg[k].append([a])
                    else:
                        test = False
                    included = True
                    break
        if included:
            key = k
            if not test:
                return cfg, key, test
            break

    # replace all instances of a and b with key
    new_g = {}
    for k in cfg:
        new_alts = []
        new_g[k] = new_alts
        for rule in cfg[k]:
            new_rule = [key if (token in {a, b} and k != token and not k.endswith('_>')) else token for token in rule]
            # if the current token is a or b, and is not the current key, then replace that token with the new key.
            # This way we equate the non-terminals a and b. See example in Section 5.
            new_alts.append(new_rule)

    if not included:
        rules = ([[a]] + [[b]])  # Make grammar compact.
        defs = {str(r): r for r in rules}
        new_g[key] = [defs[l] for l in defs]
    return new_g, key, test


def consider_merging(a, b, key, cfg, start):
    global UNMERGED_GRAMMAR
    g, key, test = gen_new_grammar(a, b, key, cfg)
    if not test:
        return False

    nodes = [a, b]
    for i in range(2):
        tk = cfg[nodes[i]][0][1]
        if i == 0:
            sg = change_nonterminal(a, b, UNMERGED_GRAMMAR)
        else:
            sg = change_nonterminal(b, a, UNMERGED_GRAMMAR)
        fzz = fuzz.CheckFuzzer(sg, nodes[i], tk)
        v = fzz.fuzz(start)
        r = check.check(v)
        if not r:
            return None
    # Merge checks passed.
    return g


# the phase_3 is merging of keys
# The keys are unordered pairs of repetition keys A'_i, A'_j which corresponds
# to repetition subexpressions
def phase_3(cfg, start):
    global UNMERGED_GRAMMAR
    UNMERGED_GRAMMAR = cfg
    # first collect all reps
    repetitions = [k for k in cfg if k.endswith('_rep>')]
    i = 0
    for (a, b) in itertools.combinations(repetitions, 2):
        c = to_key([i], '_')
        res = consider_merging(a, b, c, cfg, start)
        if res:
            cfg = res
            i += 1
        else:
            continue
    return cfg


def main():
    # phase 1
    inputs = []
    regexes = []

    # We read inputs from a file.
    with open('inputs') as f:
        inputs = [line.strip() for line in f]

    if len(inputs) == 0:
        print("inputs file is empty! Please provide inputs.")
        sys.exit()
    for input in inputs:
        regexes.append(phase_1([i for i in input]))
        print("One regex done")
    print("\n+++++ Phase 1 Done +++++\n")
    # Combine regexes into one regex as explained in Section 6.1
    regex = regexes[0]
    regexes.pop(0)
    for reg in regexes:
        regex = Alt(regex, reg, False)

    print("Final regex: " + str(regex))

    cfg, start = phase_2(regex)

    with open('grammar_.json', 'w+') as f:
        json.dump({'<start>': [[start]], **cfg}, indent=4, fp=f)

    print('\n+++++ Merging Phase Begins +++++\n')
    merged = phase_3(cfg, start)

    # Save the final grammar in the fuzzing book format
    with open('grammar.json', 'w+') as f:
        json.dump({'<start>': [[start]], **merged}, indent=4, fp=f)


if __name__ == '__main__':
    # we assume check is modified to include the
    # necessary oracle
    main()
