
# How many times to check that the nonterminals are unifiable during phase 3
P3Check = 10


# What samples to use for a{n} to conirm that a* is a valid regex.
# 2 checks per Rep only. See Section 4.2, page 5.
SAMPLES_FOR_REP = [0, 2]


# How many times to fuzz to verify
FuzzVerify = 100
