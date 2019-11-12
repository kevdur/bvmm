#===============================================================================
# BVMM
# Maximum Likelihood Functions
#===============================================================================

import math
import numpy as np
from . import tree
from . import likelihood

def mlhd(data, alphabet, runs, alpha=None, prior='uniform', complete=False,
        height_step=1, kind='sequence'):
    '''
    Estimates the maximum likelihood or a posteriori tree for a given data set.

    Args:
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions. Will be
            initialised to an array of ones by default.
        runs: the number of times the optimisation heuristic should be executed;
            the best result is returned.
        prior: the distribution to use as a prior on tree size, one of
            'uniform', 'inverse' (1/k), and 'poisson' (1/k!).
        complete: whether or not the tree should be interpreted as a complete
            tree, in which case leaves will be viewed as internal nodes, and
            their inactive children treated as leaves.
        height_step: the number of levels that should be added each time the
            tree's limits are reached and it needs to be extended with new,
            inactive nodes.
        kind: the data type, either 'sequence' or 'network'.

    Returns:
        The root of the estimated tree.
    '''
    alpha = likelihood._verify_alpha(alpha, alphabet)
    opts = tree.Options(complete, height_step=height_step, kind=kind)
    lpr = likelihood._prior_function(prior)
    ml, mroot = None, None
    for r in range(runs):
        l, root = _mlhd(data, alphabet, alpha, lpr, opts)
        if ml is None or l > ml:
            ml, mroot = l, root
    return mroot

def _mlhd(data, alphabet, alpha, lprior_ratio, opts):
    # This function attempts to find a tree of maximum likelihood by activating
    # nodes randomly until an increase in likelihood is no longer possible.
    root = tree.create_tree(opts.height_step, data, alphabet, opts.kind)
    if not opts.complete:
        tree.activate(root, data, alphabet, opts)
    l, increased = 1, True
    while increased:
        increased = False
        for a in np.random.permutation(root.attachment_count):
            v = tree.attachment(root, a)
            nc, ac = root.node_count, root.attachment_count
            if opts.complete:
                vc = sum(w.counts is not None for w in v.children)
                lr = likelihood.complete_lbirth_ratio(v, alpha) + \
                     lprior_ratio(nc+ac, nc+ac+vc)
            else:
                lr = likelihood.lbirth_ratio(v, alpha) + lprior_ratio(nc, nc+1)
            if lr > 0:
                tree.activate(v, data, alphabet, opts)
                increased = True
                l += lr
                break
    return l, root