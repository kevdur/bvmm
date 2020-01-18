#===============================================================================
# BVMM
# Markov Chain Monte Carlo Functions
#===============================================================================

import math
import numpy as np
from . import tree
from . import likelihood

class Counts:
    '''
    Tracks move counts for an MCMC run.
    '''
    def __init__(self):
        self.moves = 0
        self.skips = 0
        self.births = 0
        self.birth_attempts = 0
        self.deaths = 0
        self.death_attempts = 0

    def __str__(self):
        b, ba = self.births, self.birth_attempts
        d, da = self.deaths, self.death_attempts
        bf, df = 0 if b == 0 else b/ba, 0 if d == 0 else d/da
        return ('births: {}/{} ({:.0%})\n'
                'deaths: {}/{} ({:.0%})\n'
                'skips:  {}').format(b, ba, bf, d, da, df, self.skips)

def mcmc(data, alphabet, samples, period=1, min_skip_prob=0.1, alpha=None,
        prior='poisson', full=False, fringe=False, height_step=1,
        kind='sequence'):
    '''
    Samples trees according to their likelihoods using Markov chain Monte Carlo.

    Args:
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
        samples: the number of MCMC samples to generate.
        period: the number of MCMC moves to perform between consecutive samples.
        min_skip_prob: the base probability of proposing the 'identity' move,
            which does not alter the active tree at all. The effective
            probability will be greater whenever birth or death moves are
            impossible.
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
        The root of a tree in which each node's sample count reflects the number
        of samples it was active for, and a `Counts` object that contains move
        statistics for the MCMC run. Each nodes' sample counts give an estimate
        of the probability that its associated state was present in the model
        that generated the data.
    '''
    counts = Counts()
    alpha = likelihood._verify_alpha(alpha, alphabet)
    opts = tree.Options(full, fringe, height_step, min_skip_prob, kind)
    lpr = likelihood._prior_function(prior)
    root = tree.create_tree(height_step, data, alphabet, kind)
    if not full:
        tree.activate(root, data, alphabet, opts)
    for s in range(samples*period):
        nc, ac = root.node_count, root.attachment_count
        birth_move, death_move = _move_probs(nc, ac, opts)
        m = np.random.rand()
        if m < birth_move:
            counts.birth_attempts += 1
            if _birth(root, data, alphabet, alpha, lpr, opts):
                counts.births += 1
        elif m < birth_move + death_move:
            counts.death_attempts += 1
            if _death(root, alpha, lpr, opts):
                counts.deaths += 1
        else:
            counts.skips += 1
        if (s+1) % period == 0:
            tree.update_sample_counts(root, 1, opts=opts)
    likelihood._scale_sample_counts(root, 1/samples)
    while root.node_count > (0 if opts.full else 1):
        tree.deactivate(tree.leaf(root, 0))
    return root, counts

def _birth(root, data, alphabet, alpha, lprior_ratio, opts):
    '''
    Attempts a birth move, returning true if the move was accepted.
    '''
    # When attempting a birth move, we actually activate the proposed node by
    # default, and only deactivate it if the move is rejected. This isn't ideal,
    # but it is necessary, since we require counts that might only be available
    # after the node's children have been initialised (during activation).
    v = tree.attachment(root, np.random.randint(root.attachment_count))
    tree.activate(v, data, alphabet, opts)
    death_prob = _death_prob(v, root, alpha, lprior_ratio, opts)
    if death_prob != 0 and np.random.rand() > 1/death_prob:
        tree.deactivate(v)
        return False
    return True

def _death(root, alpha, lprior_ratio, opts):
    '''
    Attempts a death move, returning true if the move was accepted.
    '''
    v = tree.leaf(root, np.random.randint(root.leaf_count))
    death_prob = _death_prob(v, root, alpha, lprior_ratio, opts)
    if np.random.rand() <= death_prob:
        tree.deactivate(v)
        return True
    return False

def _death_prob(v, root, alpha, lprior_ratio, opts):
    '''
    Returns the acceptance probability of a death move involving a given node.

    Note that the probability of the corresponding birth move is the inverse of
    this death probability.
    '''
    nc, lc, ac = root.node_count, root.leaf_count, root.attachment_count
    nac = ac-v.attachment_count+1
    if opts.full:
        lr = likelihood.full_ldeath_ratio(v, alpha)
        pr = lprior_ratio(nc+ac, nc-1+nac)
    else:
        lr = likelihood.ldeath_ratio(v, alpha)
        pr = lprior_ratio(nc, nc-1)
    dm = _move_probs(nc, ac, opts)[1]
    bm = _move_probs(nc-1, nac, opts)[0]
    return math.exp(lr+pr)*(bm/nac)*(lc/dm)

def _move_probs(node_count, attachment_count, opts):
    '''
    Returns the probabilities of proposing birth and death moves, respectively.
    '''
    move_prob = 1-opts.min_skip_prob
    if node_count == (0 if opts.full else 1) and attachment_count == 0:
        return 0, 0
    elif node_count == (0 if opts.full else 1):
        return move_prob, 0
    elif attachment_count == 0:
        return 0, move_prob
    else:
        return .5*move_prob, .5*move_prob