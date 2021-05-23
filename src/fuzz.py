# ### Fuzzer

import random
import config

# +
class Fuzzer:
    def __init__(self, grammar):
        self.grammar = grammar

    def fuzz(self, key='<start>', max_num=None, max_depth=None):
        raise NotImplemented()

COST = None

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

    def fuzz(self, key='<start>', max_depth=10):
        return self.gen_key(key=key, depth=0, max_depth=max_depth)

    def __init__(self, grammar):
        global COST
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
        if val:
            print("Value: " + val)
            correct += 1
    print('Fuzz:', correct, '/', total)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
