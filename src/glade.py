#!/usr/bin/env python
import copy
import itertools
import json
import random
import sys

import check
import config
import fuzz

UNMERGED_GRAMMAR = {}


class Regex:
    def to_rules(self):
        if isinstance(self, Alt):
            if self.newly_generalized:
                yield from self.a1.to_rules()
                yield from self.a2.to_rules()
                self.newly_generalized = False
            elif self.a1_gen or newly_generalized_descendant(self.a1):  # Expand the first alternative if it, or one of its descendants were newly generalized.
                yield from self.a1.to_rules()
            elif self.a2_gen or newly_generalized_descendant(self.a2):  # Expand the second alternative if it, or one of its descendants were newly generalized.
                yield from self.a2.to_rules()

            else:  # Else it's part of the context. We don't enumerate all rules, but pick randomly only one of the two components.
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

        elif isinstance(self, String):
            assert not isinstance(self.o, Regex)
            yield self.o[-1]  # return last added character, or the original character if none were added.
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
        elif isinstance(self, String):
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
        return "(%s)" % ' '.join(repr(a) for a in self.arr if a)


class Alts(Regex):
    def __init__(self, arr):
        self.arr = arr

    def __repr__(self):
        return "(%s)" % ' | '.join(repr(a) for a in self.arr if a)


class String(Regex):
    def __init__(self, o, extra, generalized=0, curr_char_gen=False):
        self.o = o  # A list containing the original character and all possible character replacements.
        self.next_gen = extra  # Substrings are annotated with extra data to express possible further generalization options.
                               # 0: for no generalization, 1: for Rep, 2: for Alt.
                               # Section 4.1: These annotations indicate that the
                               # bracketed substring in the current regular expression can be
                               # generalized by adding either a repetition (if tau = rep) or an
                               # alternation (if tau = alt).
        self.generalized = generalized   # 127 if all possible character replacements have been tried.
        self.curr_char_gen = curr_char_gen  # True if it's the current terminal being generalized in the Char Generalization Phase.

    def __repr__(self):
        return "(%s)" % ' '.join(repr(a) for a in self.o if a)


# Alternations: If generalizing P alt[alpha]Q, then
# for each decomposition alpha = a_1 a_2, where a_1 != [] and
# a_2 != [], generate P (rep[alpha_1] + alt[alpha_2]) Q
# ...
# in both cases, P alpha Q is also generated
# + is alternation i.e `|' in regular expression

# Ordering: If generalizing P alt[alpha] Q we prioritize shorter
# alpha_1.
# In either case, P alpha Q is ranked last
# Note that candidate repetitions and candidate alternations can
# be ordered independently
# We don't generalize all descendants in one go, but only one substring at each step. Section 4.1 page 4
# each generalization step selects a single bracketed substring [\alpha]\tau and generates candidates based on decompositions of \alpha


def gen_alt(alpha):
    length = len(alpha)
    for i in range(1, length):  # shorter alpha_1 prioritized
        alpha_1, alpha_2 = alpha[:i], alpha[i:]

        # alpha_1 != epsilon and alpha_2 != epsilon
        assert alpha_1
        assert alpha_2

        yield Alt(String([alpha_1], 1), String([alpha_2], 2), True)

    if length:  # this is the final choice.
        # There is an inconsistency between the Figure 2 and the Section 4.1
        # We chose to follow the Figure 2.
        yield String([alpha], 1)
        # The text requires this
        # yield String([alpha], 0)
        # However, we note that there is no functionality difference between
        # both.

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
        yield String([alpha], 0)
    else:
        for i in range(length):  # shorter alpha1 prioritized
            alpha_1 = alpha[:i]
            for k in range(i + 1, length + 1):  # longer alpha2 prioritized, see section 4.2
                j = length - (k - (i + 1))      # j is the inverse of k.
                alpha_2, alpha_3 = alpha[i:j], alpha[j:]

                # alpha_2 != epsilon
                assert alpha_2

                if i == 0 and j == length:
                    yield Rep(String([alpha_2], 2), True)
                elif i != 0 and j != length:
                    yield Seq([String([alpha_1], 0), Rep(String([alpha_2], 2), True), String([alpha_3], 1)])
                elif i == 0 and j != length:
                    yield Seq([Rep(String([alpha_2], 2), True), String([alpha_3], 1)])
                elif i != 0 and j == length:
                    yield Seq([String([alpha_1], 0), Rep(String([alpha_2], 2), True)])
        if length:  # the final choice
            yield String([alpha], 0)
    return


# List of all printable ASCII characters.
all_chars = [chr(i) for i in range(128)]


def gen_char(regex):
    # This function transforms a regex in-place by incrementally adding
    # alternative terminal characters. After each insertion, the modified
    # terminal is yield to give a chance to the calling process to check and
    # potentially rollback the change.

    if isinstance(regex, Rep):
        regex.newly_generalized = False
        yield from gen_char(regex.a)

    elif isinstance(regex, Alt):
        regex.newly_generalized = False
        regex.a1_gen = True
        yield from gen_char(regex.a1)

        regex.a1_gen = False
        regex.a2_gen = True
        yield from gen_char(regex.a2)

        regex.a1_gen = False
        regex.a2_gen = False

    elif isinstance(regex, Seq):
        for obj in regex.arr:
            yield from gen_char(obj)

    elif isinstance(regex, String):
        while regex.generalized < len(all_chars) - 1:
            # perform a generalization step.
            regex.curr_char_gen = True
            curr_char = all_chars[regex.generalized]
            if curr_char == regex.o[0]:
                # skip adding this character since it is the initial one
                regex.generalized += 1
                curr_char = all_chars[regex.generalized]

            regex.o.append(curr_char)
            regex.generalized += 1
            yield regex

        # All chars have been tried, we mark current unit as non-generalizable.
        regex.curr_char_gen = False


def atomize(regex):
    # Explode String-regexes into sequences of String-regexes of one character.
    # e.g. ("abc")* -> ("a" "b" "c")*

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

    elif isinstance(regex, String):
        regex_orig = copy.deepcopy(regex)
        stg = regex.o[0]
        if len(stg) > 1:
            regex.o.pop()
            return Seq([String([i], 0) for i in stg])

        return String(regex_orig.o[0], 0)


def newly_generalized_descendant(regex):
    # Check if one descendant has been newly generalized.
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
        return any(newly_generalized_descendant(obj) for obj in regex.arr)

    elif isinstance(regex, String):
        return False


def linearize_rep(regex):
    # Linearize nested Rep in regex. e.g. ((((a)*)*)* | b) -> ((a)* | b)

    if isinstance(regex, Rep):
        if isinstance(regex.a, Rep):
            regex = regex.a
            return linearize_rep(regex)
        else:
            child = linearize_rep(regex.a)
            return Rep(child, regex.newly_generalized)

    elif isinstance(regex, Alt):
        regex.a1 = linearize_rep(regex.a1)
        regex.a2 = linearize_rep(regex.a2)
        return regex

    elif isinstance(regex, Seq):
        i = 0
        for obj in regex.arr:
            obj = linearize_rep(obj)
            regex.arr[i] = obj
            i += 1
        return regex

    elif isinstance(regex, String):
        return regex


def linearize_alt(regex):
    # Linearize nested Alt in regex. e.g. ((a|(b|c)) d) -> ((a|b|c) d)

    if isinstance(regex, Rep):
        regex.a = linearize_alt(regex.a)
        return regex

    elif isinstance(regex, Alt):
        e1 = linearize_alt(regex.a1)
        e2 = linearize_alt(regex.a2)
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
            obj = linearize_alt(obj)
            regex.arr[i] = obj
            i += 1
        return regex

    elif isinstance(regex, String):
        return regex


def character_generalization_phase(regex):
    # Character generalization phase that generalizes terminals in the
    # synthesized regular expression `regex`. The algorithm considers
    # generalizing each terminal in the regex to every (different) terminal in
    # Sigma.
    #
    # See Section 6.2 Page 8.

    for generalization_attempt in gen_char(regex):
        exprs = list(to_strings(regex))
        if not all(check.check(expr, regex) for expr in exprs):
            # rollback the last change
            del generalization_attempt.o[-1]

    return regex


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
NON_GENERALIZABLE = -1


# The get_candidates function is the generator of candidates. It's called at each step once, it selects a terminal substring
# then generates all possible generalization for that substring. Each representing a candidate regex.
def get_candidates(regex):
    exp = False  # Used to insure that we don't modify more that one branch in each step.
    if isinstance(regex, Rep):
        for x in get_candidates(regex.a):
            if x == NON_GENERALIZABLE:  # We reached a leaf that is non-generalizable.
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

    elif isinstance(regex, String):
        if regex.next_gen == 0:
            yield NON_GENERALIZABLE
        elif regex.next_gen == 1:
            yield from gen_rep(regex.o[-1])
            regex.next_gen = 0
        elif regex.next_gen == 2:
            yield from gen_alt(regex.o[-1])
            regex.next_gen = 0


# This helper function is here only to help print the regex hierarchy.
def get_dict(regex):
    if isinstance(regex, Rep):
        return {"Rep": [get_dict(regex.a), regex.newly_generalized]}
    elif isinstance(regex, Alt):
        return {"Alt": [get_dict(regex.a1) , get_dict(regex.a2), regex.newly_generalized]}
    elif isinstance(regex, Seq):
        return {"Seq": [get_dict(obj) for obj in regex.arr]}
    elif isinstance(regex, String):
        regex.o.insert(0, str(regex.next_gen))
        return {"String": regex.o}
    else:
        return "Nothing to return!"


def phase_1(alpha_in):
    # Active learning of regular right-hand side from Bastani et al.
    #
    # Each generalization step selects a single bracketed substring
    # T[alpha] and generates candidates based on decompositions of alpha
    # i.e. an expression of alpha as alpha = a_1, a_2, ..a_k
    #
    # Each iteration of the while loop corresponds to one generalization step.
    # The code below follows Algorithm 1, page 3 in the paper.

    # Seed input alpha_in is annotated rep(alpha_in)
    curr_reg = String([alpha_in], 1)

    done = False
    while not done:
        started = False
        # The get_candidates function supplies candidates, and is equivalent to the function "ConstructCandidates()" in the paper.
        for regex in get_candidates(curr_reg):
            started = True
            if regex == NON_GENERALIZABLE:
                # No more generalizations are possible. We are done with Phase 1.
                done = True
                break

            regex = linearize_rep(regex)
            # to_strings() function is equivalent to the function ConstructChecks() in the paper.
            exprs = list(to_strings(regex))

            ay = copy.deepcopy(regex)
            ay = linearize_rep(ay)
            var = str(get_dict(ay))
            if var in valid_regexes:
                continue

            if str(regex) in regex_map:
                all_true = regex_map[str(regex)]
            else:
                all_true = all(check.check(expr, regex) for expr in exprs)

            regex_map[str(regex)] = all_true

            if all_true:
                # we found the candidate for the next generalization step
                ayy = copy.deepcopy(regex)
                var = str(get_dict(ayy))
                valid_regexes.add(var)
                curr_reg = regex
                break

        if not started:
            break

    # Before executing the Character Generalization Phase, we break strings in
    # regex into separate chars, that is, given a String-regex, we break it into
    # a Seq-regex that contains String-regexes each containing a single char.
    # This way we can systematically generalize each char/terminal/sigma_i
    # separately.
    atomized_reg = atomize(curr_reg)

    final_reg = character_generalization_phase(atomized_reg)
    return linearize_alt(final_reg)


def to_key(prefix, suffix=''):
    return '<k%s%s>' % (''.join(str(s) for s in prefix), suffix)


# if step i generalizes P rep[alpha] Q to
# P alpha_1 (alt[alpha_2])* rep[alpha_3] Q
# we generate productions
# A_i -> alpha_1 A'_i A_k
# A'i -> \e + A'_i A_j
# equivalent to A_i -> alpha_1 A_j* A_k
# where A_k comes from rep[alpha_3] and
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


def extract_string(regex, prefix):
    if len(regex.o) == 1:  # string is a terminal character
        return {}, ''.join(regex.o[0])
    else:  # string is a non terminal, meaning it has been generalized to a list of n chars. Therefore we treat it as an Alt object with n alternatives. See example in section 6.2
        return {to_key(prefix): [[t] for t in regex.o]}, to_key(prefix)


def phase_2(regex):
    # The basic idea is to first translate the regexp into a
    # CFG, where the terminal symbols are the symbols in the
    # regex, and the generalization steps are non-terminals
    # and next, to equate the non-terminals in that grammar
    # to each other
    # Alt, Rep, Seq, String
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
    elif isinstance(regex, String):
        return extract_string(regex, prefix)
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
            # If the current token is a or b, and is not the current key, then replace that token with the new key.
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


# The phase_3 is merging of keys
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

    with open('inputs') as f:
        inputs = [line.strip() for line in f]

    if len(inputs) == 0:
        print("inputs file is empty! Please provide inputs.")
        sys.exit()

    for input_str in inputs:
        regexes.append(phase_1([i for i in input_str]))
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
    # we assume check is modified to include the necessary oracle
    main()
