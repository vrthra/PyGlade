This is an implementation of the _Glade_ blackbox grammar miner described by
Bastani et al. in [Synthesizing Program Input
Grammars](https://arxiv.org/pdf/1608.01723.pdf)

To use, modify the `src/check.py` file which contains the oracle. Controlling
the number of attempts to verify is done in `src/config.py`

Seed inputs are placed in the `inputs` file.

To learn the grammar, execute:

    make gen

To generate inputs from the learned grammar, run:

    make fuzz


