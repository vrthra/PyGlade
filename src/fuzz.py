# ### Fuzzer

import random
import config
import pprint
# +
class Fuzzer:
    def __init__(self, grammar):
        self.grammar = grammar

    def fuzz(self, key='<start>', max_num=None, max_depth=None):
        raise NotImplemented()

COST = None

# CheckFuzzer class implements the check construction for merges
# For each merge, two checks are constructed, follwoing section 5.3.
# The new key recently added into a grammar is assigned a cost of -1,
# ie: the least cost, that is to insure that it gets expanded first.
class CheckFuzzer(Fuzzer):
    def symbol_cost(self, grammar, symbol, seen):
        if symbol in self.key_cost: return self.key_cost[symbol]
        if symbol in seen:
            self.key_cost[symbol] = float('inf')
            return float('inf')
        if symbol == self.new_key:
            v = -1
        else:
            v = min((self.expansion_cost(grammar, rule, seen | {symbol})
                        for rule in grammar.get(symbol, [])), default=0)
        self.key_cost[symbol] = v
        return v

    def expansion_cost(self, grammar, tokens, seen):
        if min((self.symbol_cost(grammar, token, seen) for token in tokens if token in grammar), default=0) == -1:
            return -1
        else:
            return max((self.symbol_cost(grammar, token, seen)
                        for token in tokens if token in grammar), default=0) + 1

    def gen_key(self, key, depth, max_depth):
        if key not in self.grammar: return key
        if key == self.new_key:
            rules = [self.grammar[key][self.alt]]
            self.alt = -1

        elif key == self.a_key:
            if self.a_check == 2:
                rules = [self.grammar[key][1]]
            else:
                rules = [self.grammar[key][0]]
                self.a_check += 1

        elif key == self.b_key:
            if self.b_check == 2:
                rules = [self.grammar[key][1]]
            else:
                rules = [self.grammar[key][0]]
                self.b_check += 1

        else:
            pathl = [(key, rule) for rule in self.grammar[key] if self.cost[key][str(rule)] == -1]
            if len(pathl) > 1:
                clst = sorted([(self.normal_cost[k][str(r)], r) for k, r in pathl])
            else:
                clst = sorted([(self.cost[key][str(rule)], rule) for rule in self.grammar[key]])
            rules = [r for c,r in clst if c == clst[0][0]]

        chosen_rule = random.choice(rules)
        current_expansion = key + ''.join(chosen_rule)
        if key.endswith('_rep>') and key != self.a_key and key != self.b_key and current_expansion in self.past_expansions: 
            if clst[0][0] == -1:
                rules = [r for c,r in clst if c == clst[1][0]]
                chosen_rule = random.choice(rules)
                current_expansion = key + ''.join(chosen_rule)

        self.past_expansions.add(current_expansion)
        return self.gen_rule(chosen_rule, depth+1, max_depth)

    def gen_rule(self, rule, depth, max_depth):
        return ''.join(self.gen_key(token, depth, max_depth) for token in rule)

    def fuzz(self, key='<start>', max_depth=100):
        self.past_expansions = set()
        return self.gen_key(key=key, depth=0, max_depth=max_depth)

    def alt_pos(self):
        x = self.grammar[self.new_key].index([self.a_key])
        if x != (len(self.grammar[self.new_key]) - 1): return x
        else: return self.grammar[self.new_key].index([self.b_key])

    def __init__(self, grammar, new_key, a_key, b_key):
        global COST
        super().__init__(grammar)
        self.new_key = new_key
        self.a_key = a_key
        self.b_key = b_key
        self.a_check = 0
        self.b_check = 0
        self.past_expansions = set()
        self.alt = self.alt_pos() # First alternative index to try.
        self.key_cost = {}
        COST = self.compute_cost(grammar)
        self.cost = COST
        self.normal_cost = LimitFuzzer(grammar).cost

    def compute_cost(self, grammar):
        cost = {}
        for k in grammar:
            cost[k] = {}
            for rule in grammar[k]:
                try:
                    cost[k][str(rule)] = self.expansion_cost(grammar, rule, set())
                except Exception as e:
                    cost[k][str(rule)] = float('inf')
        return cost

class LimitFuzzer(Fuzzer):
    def symbol_cost(self, grammar, symbol, seen):
        if symbol in self.key_cost: return self.key_cost[symbol]
        if symbol in seen:
            self.key_cost[symbol] = float('inf')
            return float('inf')
        v = min((self.expansion_cost(grammar, rule, seen | {symbol})
                    for rule in grammar.get(symbol, [])), default=0)
        self.key_cost[symbol] = v
        return v

    def expansion_cost(self, grammar, tokens, seen):
        return max((self.symbol_cost(grammar, token, seen)
                    for token in tokens if token in grammar), default=0) + 1

    def gen_key(self, key, depth, max_depth):
        if key not in self.grammar: return key
        if depth > max_depth:
            clst = sorted([(self.cost[key][str(rule)], rule) for rule in self.grammar[key]])
            rules = [r for c,r in clst if c == clst[0][0]]
        else:
            rules = self.grammar[key]
        return self.gen_rule(random.choice(rules), depth+1, max_depth)

    def gen_rule(self, rule, depth, max_depth):
        return ''.join(self.gen_key(token, depth, max_depth) for token in rule)

    def fuzz(self, key='<start>', max_depth=100):
        return self.gen_key(key=key, depth=0, max_depth=max_depth)

    def __init__(self, grammar):
        global counter
        super().__init__(grammar)
        self.key_cost = {}
        COST = self.compute_cost(grammar)
        self.cost = COST

    def compute_cost(self, grammar):
        cost = {}
        for k in grammar:
            cost[k] = {}
            for rule in grammar[k]:
                try:
                    cost[k][str(rule)] = self.expansion_cost(grammar, rule, set())
                except Exception as e:
                    cost[k][str(rule)] = float('inf')
        return cost

import json
import check
def main(fn):
    with open(fn) as f:
        mgrammar = json.load(fp=f)
    fuzzer = LimitFuzzer(mgrammar)
    correct = 0
    total = config.FuzzVerify
    for i in range(total):
        val = fuzzer.fuzz(mgrammar['<start>'][0][0])
        #val = fuzzer.fuzz(mgrammar['[grammar]']['<start>'][0][0])
        correc = check.check(val)
        if correc:
            print("Correct Value: " + val)
            correct += 1
        else:
            print("Incorrect Value: " + val)
    print('Fuzz:', correct, '/', total)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
