# Variable-order Markov Models and Bayesian Model Selection

This is a relatively straightforward application of Bayesian inference using
conjugate priors and Markov chain Monte Carlo (for handling parameter and model
inference respectively) to simple, sequential variable-order Markov models,
which can be naturally represented as context trees.

The point of the exercise is to uncover dependencies among symbols in sequential
data sets, which is done by sampling from the posterior model distribution and
aggregating the dependencies observed in each model. The results are
interesting, but admittedly more from an interest's-sake perspective than one of
practical application.

See the [write-up](doc/bvmm.pdf) for full details.

## Project structure

The code is written in Python, and stored in the `bvmm` directory. There are
examples of how to use it (and how the data sets were generated and processed)
in the `examples` directory.

The write-up, which contains all of the motivational, theoretical, and
implementation details, as well as a summary of results, is available
[here](doc/bvmm.pdf), in the `doc` directory.
