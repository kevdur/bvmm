#===============================================================================
# BVMM
# Probability and Likelihood Functions
#===============================================================================

import math
import numpy as np
from . import tree
from scipy.special import gammaln

def _verify_alpha(alpha, alphabet):
    if alpha is None:
        return np.ones(len(alphabet))
    elif len(alpha) != len(alphabet):
        raise ValueError('The alpha array must have the same length as the '
                         'alphabet array.')
    else:
        return alpha

def _prior_function(prior):
    if prior.lower() == 'uniform':
        return luniform_ratio
    elif prior.lower() == 'inverse':
        return linverse_ratio
    elif prior.lower() == 'poisson':
        return lpoisson_ratio
    else:
        raise ValueError("Invalid prior distribution specified. Valid options "
                         "are 'uniform', 'inverse', and 'poisson'.")

def luniform_ratio(k, l):
    '''
    Returns the logarithm of the uniform prior ratio, which is simply 0.

    This function (and in part `linverse_ratio`) are here simply because they
    lead to cleaner code.

    Args: k, l: unused.
    '''
    return 0

def linverse_ratio(k, l):
    '''
    Returns the logarithm of p(l)/p(k), where p(x) is the size prior 1/x.

    Args:
        k: the size of the current tree.
        l: the size of the proposed tree.
    '''
    return math.log(k) - math.log(l)

def lpoisson_ratio(k, l):
    '''
    Returns the logarithm of p(l)/p(k), where p(x) is the size prior 1/x!.

    Args:
        k: the size of the current tree.
        l: the size of the proposed tree.
    '''
    if k > l:
        return sum(math.log(x) for x in range(l+1, k+1))
    else:
        return sum(-math.log(x) for x in range(k+1, l+1))

def lbirth_ratio(v, alpha):
    '''
    Returns the log-likelihood ratio for a proposed birth move.

    Args:
        v: the inactive attachment node that is to be activated (possibly).
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions.
    '''
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

def ldeath_ratio(v, alpha):
    '''
    Returns the log-likelihood ratio for a proposed death move.

    Args:
        v: the active leaf node that is to be deactivated (possibly).
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions.
    '''
    return -lbirth_ratio(v, alpha)

def full_lbirth_ratio(v, alpha):
    '''
    Returns the log-likelihood ratio for a proposed child-complete birth move.

    Args:
        v: the attachment node (which is viewed as a leaf when treating the tree
            as full) which is to be activated (possibly).
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions.
    '''
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

def full_ldeath_ratio(v, alpha):
    '''
    Returns the log-likelihood ratio for a proposed child-complete death move.

    Args:
        v: the leaf node (which is viewed as an internal node when treating the
            tree as full) which is to be deactivated (possibly).
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions.
    '''
    return -full_lbirth_ratio(v, alpha)

def _llhd(root, data, alphabet, alpha, lprior_ratio, opts):
    '''
    Returns the unnormalised log-likelihood of a given tree.

    The log-likelihood value is given relative to that of the singleton tree
    (consisting of nothing but a root), which is assigned a log-likelihood of 0.

    Args:
        lprior_ratio: the ratio function of the model's prior, either
            `luniform_ratio`, `linverse_ratio`, or `lpoisson_ratio`.
    '''
    # This algorithm proceeds by deactivating leaves, recording the change in
    # likelihood after each step.
    l, vs = 0, []
    while root.node_count > (0 if opts.full else 1):
        v = tree.leaf(root, 0)
        nc, ac, vc = root.node_count, root.attachment_count, v.attachment_count
        if opts.full:
            l -= full_ldeath_ratio(v, alpha) + lprior_ratio(nc+ac, nc+ac-vc)
        else:
            l -= ldeath_ratio(v, alpha) + lprior_ratio(nc, nc-1)
        tree.deactivate(v)
        vs.append(v)
    for v in reversed(vs):
        tree.activate(v, data, alphabet, opts)
    return l

def bf(data, alphabet, max_height=np.inf, alpha=None, prior='uniform', 
        full=False, fringe=False, height_step=1, kind='sequence'):
    '''
    Computes the probabilities of all possible Markov trees by brute force.
    
    Args:
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
        max_height: only those trees whose height is less than or equal to the
            given value will be considered.
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions. Will be
            initialised to an array of ones by default.
        prior: the distribution to use as a prior on tree size, one of
            'uniform', 'inverse' (1/k), and 'poisson' (1/k!).
        full: whether or not the tree should be interpreted as a full tree, in
            which case leaves will be viewed as internal nodes, and their
            inactive children treated as leaves.
        fringe: if true, strict internal nodes (whose children are all active)
            will not have their sample counts updated.
        height_step: the number of levels that should be added each time the
            tree's limits are reached and it needs to be extended with new,
            inactive nodes.
        kind: the data type, either 'sequence' or 'network'.
    
    Returns:
        The root of a tree in which each node's sample count reflects the
        probability that its associated state was present in the model that
        generated the data.
    '''
    alpha = _verify_alpha(alpha, alphabet)
    opts = tree.Options(full, fringe, height_step, kind=kind)
    lpr = _prior_function(prior)
    root = tree.create_tree(height_step, data, alphabet, kind)
    if not full:
        tree.activate(root, data, alphabet, opts)
    lsm = 0
    for s in _subtrees(root, 0, max_height, data, alphabet, opts):
        lhd = math.exp(_llhd(root, data, alphabet, alpha, lpr, opts))
        lsm += lhd
        tree.update_sample_counts(root, lhd, opts=opts)
    _scale_sample_counts(root, 1/lsm)
    return root

def _subtrees(v, i, h, data, alphabet, opts):
    '''
    Generates all possible subtrees of a given tree.

    Args:
        v: the root of the tree whose subtrees should be generated. If this node
            is already active, the empty subtree (in which it is not active)
            will not be yielded.
        i: the index of the first child to consider when generating subtrees. If
            0, all possible subtrees will be generated; otherwise, all possible
            combinations of subtrees from those children of `v` whose indices
            are greater than or equal to `i` are generated. (In this way
            subtrees can be generated recursively, regardless of the arity of
            the tree.)
        h: the maximum height of the generated subtrees.

    Yields:
        Each time this function yields (returning nothing), the active tree
        rooted at `v` will be one of its possible subtrees.
    '''
    if not v.is_active and i == 0:
        yield
        if v.attachment_count > 0 and h >= (1 if opts.full else 0):
            tree.activate(v, data, alphabet, opts)
    if v.is_active:
        if i == len(alphabet) - 1:
            for s in _subtrees(v.children[i], 0, h-1, data, alphabet, opts):
                yield
        else:
            for s in _subtrees(v.children[i], 0, h-1, data, alphabet, opts):
                for t in _subtrees(v, i+1, h, data, alphabet, opts):
                    yield
        if i == 0:
            tree.deactivate(v)

def _scale_sample_counts(root, scale):
    root.sample_count *= scale
    for w in root.children:
        _scale_sample_counts(w, scale)