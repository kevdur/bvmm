#===============================================================================
# BVMM
# Probability and Likelihood Functions
#===============================================================================

import numpy as np
from .tree import create_tree, initialise_counts, update_sample_counts, \
    leaf, activate, deactivate
from math import log
from scipy.special import gammaln

'''
Returns the logarithm of p(l)/p(k), where p(x) is the size prior p(x) = 1/x!.

Args:
    k: the size of the current tree.
    l: the size of the proposed tree.
'''
def lpoisson_ratio(k, l):
    if k > l:
        return sum(log(x) for x in range(l+1, k+1))
    else:
        return sum(-log(x) for x in range(k+1, l+1))

'''
Returns the log-likelihood ratio for a proposed birth move.

Args:
    v: the inactive attachment node that is to be activated (possibly).
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
'''
def lbirth_ratio(v, alpha):
    ucounts = np.array(v.parent.counts) # holds effective counts before birth.
    for w in v.parent.children:
        if w.node_count > 0 and w != v: # the '!=' only applies when
            ucounts -= w.counts         # calculating a death ratio.
    l = np.sum(gammaln(ucounts-v.counts+alpha)) - np.sum(gammaln(ucounts+alpha))
    l += np.sum(gammaln(v.counts+alpha)) - np.sum(gammaln(alpha))
    usm, vsm, asm = np.sum(ucounts), np.sum(v.counts), np.sum(alpha)
    l += -gammaln(usm-vsm+asm) + gammaln(usm+asm)
    l += -gammaln(vsm+asm) + gammaln(asm)
    return l

'''
Returns the log-likelihood ratio for a proposed death move.

Args:
    v: the active leaf node that is to be deactivated.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
'''
def ldeath_ratio(v, alpha):
    return -lbirth_ratio(v, alpha)

'''
Returns the log-likelihood ratio for a proposed tree-complete birth move.

Args:
    v: the active leaf node whose children are to be activated.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
'''
def complete_lbirth_ratio(v, alpha):
    ccounts = np.zeros_like(v.counts) # holds the sum of the children's counts.
    vsm, asm = np.sum(v.counts), np.sum(alpha)
    abeta = np.sum(gammaln(alpha)) - gammaln(asm)
    l = -np.sum(gammaln(v.counts+alpha)) + gammaln(vsm+asm)
    for w in v.children:
        if w.counts is not None:
            ccounts += w.counts
            l += np.sum(gammaln(w.counts+alpha)) - gammaln(np.sum(w.counts)+asm)
            l -= abeta
    csm = np.sum(ccounts)
    l += np.sum(gammaln(v.counts-ccounts+alpha)) - gammaln(vsm-csm+asm)
    return l

'''
Returns the log-likelihood ratio for a proposed tree-complete death move.

Args:
    v: the active internal node whose children (all of which are active leaves,
        if they are valid) are to be deactivated.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
'''
def complete_ldeath_ratio(v, alpha):
    return -complete_lbirth_ratio(v, alpha)

'''
Returns the unnormalised likelihood of a given tree.

The likelihood value is given relative to that of the singleton tree (consisting
of nothing but a root), which is assigned a likelihood of 1.

Args:
    root: the root of the tree whose relative likelihood is to be computed.
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
    complete: whether or not the tree is complete. Setting this to false for a
        complete tree will not cause any issues, but performance might be
        slightly worse.
'''
def lhd(root, data, alphabet, alpha, complete=False):
    # This algorithm proceeds by deactivating leaves, recording the change in
    # likelihood after each step.
    l, vs = 1, []
    while root.node_count > 1:
        if complete:
            # TODO
        else:
            v = leaf(root, 0)
            l *= ldeath_ratio(v, alpha)
            deactivate(v)
            vs.append(v)
    for v in reversed(vs):
        activate(v, data, alphabet)
    return 1/l






'''
Returns the likelihood of a given tree relative to the singleton tree.

Args:
    root: the root of the tree whose relative likelihood is to be computed.
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
'''
def lhd(root, data, alphabet, alpha):
    # This algorithm proceeds by deactivating nodes one by one, recording the
    # change in likelihood at each step.
    l, vs = 1, []
    while root.node_count > 1:
        v = leaf(root, 0)
        l *= deathratio(v, alpha)
        deactivate(v)
        vs.append(v)
    for v in reversed(vs):
        activate(v, data, alphabet)
    return 1/l









'''
Computes the posterior probabilities of each Markov state by brute force.

Args:
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
    height_step: the number of levels that should be added each time the tree's
        limits are reached and it needs to be extended with new, inactive nodes.

Returns:
    The root of a tree in which each node's sample count reflects the
    probability that its associated state was present in the model that
    generated the data.
'''
def bf(data, alphabet, alpha, height_step=1):
    root = create_tree(height_step, alphabet)
    initialise_counts(root, data, alphabet)
    activate(root, data, alphabet)
    for s in _subtrees(root.children[0], 0, data, alphabet, height_step):
        for t in _subtrees(root, 1, data, alphabet, height_step):
            update_sample_counts(root, lhd(root, data, alphabet, alpha))
    update_sample_counts(root, samples=None, scale=1/root.sample_count)
    return root

'''
Generates subtrees of a given tree by activating and deactivating nodes.

Args:
    v: the root of the tree whose subtrees should be generated. If this node is
        an actual root (i.e., it has no parent), it should already be active.
    i: the index of the first child to consider when generating subtrees. If 0,
        all possible subtrees will be generated; otherwise, all possible
        combinations of subtrees from those children of `v` whose indices are
        greater than or equal to `i` are generated. (In this way subtrees can be
        generated recursively, regardless of the arity of the tree.)

Yields:
    Each time this function yields (returning nothing), the active tree rooted
    at `v` will be one of its possible subtrees.
'''
def _subtrees(v, i, data, alphabet, height_step=1):
    if i == 0 and v.parent is not None:
        yield
        if v.counts is not None:
            activate(v, data, alphabet, height_step)
    if v.is_active:
        if i == len(alphabet) - 1:
            for s in _subtrees(v.children[i], 0, data, alphabet, height_step):
                yield
        else:
            for s in _subtrees(v.children[i], 0, data, alphabet, height_step):
                for t in _subtrees(v, i+1, data, alphabet, height_step):
                    yield
        if i == 0 and v.parent is not None:
            deactivate(v)