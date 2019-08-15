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

def mcmc(data, alphabet, alpha, samples, prior='uniform', complete=False,
        fringe=False, height_step=1, min_skip_prob=1/3):
    '''
    Samples trees according to their likelihoods using Markov chain Monte Carlo.

    Args:
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet prior on the nodes' categorical distributions.
        samples: the number of moves in the MCMC run.
        prior: the distribution to use as a prior on tree size, one of
            'uniform', 'inverse' (1/k), and 'poisson' (1/k!).
        complete: whether or not the tree should be interpreted as a complete
            tree, in which case leaves will be viewed as internal nodes, and
            their inactive children treated as leaves.
        fringe: if true, strict internal nodes (whose children are all active)
            will not have their sample counts updated.
        height_step: the number of levels that should be added each time the
            tree's limits are reached and it needs to be extended with new,
            inactive nodes.
        min_skip_prob: the base probability of proposing the 'identity' move,
            which does not alter the active tree at all. The effective
            probability will be greater whenever birth or death moves are
            impossible.

    Returns:
        The root of a tree in which each node's sample count reflects the number
        of samples it was active for, and a `Counts` object that contains move
        statistics for the MCMC run. Each nodes' sample counts give an estimate
        of the probability that its associated state was present in the model
        that generated the data.
    '''
    counts = Counts()
    lpr = likelihood.prior_function(prior)
    root = tree.create_tree(height_step, alphabet)
    tree.initialise_counts(root, data, alphabet)
    tree.activate(root, data, alphabet, complete, height_step)
    for s in range(samples):
        nc, ac = root.node_count, root.attachment_count
        birth_move, death_move = _move_probs(nc, ac, complete, min_skip_prob)
        m = np.random.rand()
        if m < birth_move:
            counts.birth_attempts += 1
            if _birth(root, data, alphabet, alpha, lpr, complete, height_step, min_skip_prob):
                counts.births += 1
        elif m < birth_move + death_move:
            counts.death_attempts += 1
            if _death(root, alpha, lpr, complete, min_skip_prob):
                counts.deaths += 1
        else:
            counts.skips += 1
        tree.update_sample_counts(root, 1, complete=complete, fringe=fringe)
    return root, counts

def _birth(root, data, alphabet, alpha, lprior_ratio, complete=False,
        height_step=1, min_skip_prob=1/3):
    '''
    Attempts a birth move, returning true if the move was accepted.
    '''
    # When attempting a birth move, we actually activate the proposed node by
    # default, and only deactivate it if the move is rejected. This isn't ideal,
    # but it is necessary, since we require counts that might only be available
    # after the node's children have been initialised (during activation).
    v = tree.attachment(root, np.random.randint(root.attachment_count))
    tree.activate(v, data, alphabet, complete, height_step)
    death_prob = _death_prob(root, alpha, lprior_ratio, complete, min_skip_prob)
    if np.random.rand() > 1/death_prob:
        tree.deactivate(v)
        return False
    return True

def _death(v, root, alpha, lprior_ratio, complete, min_skip_prob=1/3):
    '''
    Attempts a death move, returning true if the move was accepted.
    '''
    v = tree.leaf(root, np.random.randint(root.leaf_count))
    death_prob = _death_prob(root, alpha, lprior_ratio, complete, min_skip_prob)
    if np.random.rand() <= death_prob:
        tree.deactivate(v)
        return True
    return False

def _death_prob(v, root, alpha, lprior_ratio, complete=False,
        min_skip_prob=1/3):
    '''
    Returns the acceptance probability of a death move involving a given node.

    Note that the probability of the reciprocal birth move is the inverse of
    this death probability.
    '''
    nc, lc, ac = root.node_count, root.leaf_count, root.attachment_count
    nac = ac-v.attachment_count+1
    if complete:
        lr = likelihood.complete_ldeath_ratio(v, alpha)
        pr = lprior_ratio(nc+ac, nc-1+nac)
    else:
        lr = likelihood.ldeath_ratio(v, alpha)
        pr = lprior_ratio(nc, nc-1)
    dm = _move_probs(nc, ac, complete, min_skip_prob)[1]
    bm = _move_probs(nc-1, nac, complete, min_skip_prob)[0]
    return math.exp(lr+pr)*(bm/nac)*(lc/dm)

def _move_probs(node_count, attachment_count, complete=False,
        min_skip_prob=1/3):
    '''
    Returns the probabilities of proposing birth and death moves, respectively.
    '''
    move_prob = 1-min_skip_prob
    if node_count == (0 if complete else 1) and attachment_count == 0:
        return 0, 0
    elif node_count == (0 if complete else 1):
        return move_prob, 0
    elif attachment_count == 0:
        return 0, move_prob
    else:
        return .5*move_prob, .5*move_prob