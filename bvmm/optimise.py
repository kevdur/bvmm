#===============================================================================
# BVMM
# Maximum Likelihood Functions
#===============================================================================

from .likelihood import lhd, birthratio
from .tree import create_tree, initialise_counts, attachment, activate
from math import exp
from numpy.random import permutation

'''
Estimates the tree of maximal likelihood for a given data set.

Args:
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
    alpha: the 'concentration' vector that is used to parameterise the Dirichlet
        prior on the nodes' categorical distributions.
    runs: the number of times the optimisation heuristic should be executed; the
        best result is returned.
    complete: if true, only complete trees are considered, implying that a
        node's children must be activated as a group rather than individually.
    height_step: the number of levels that should be added each time the tree's
        limits are reached and it needs to be extended with new, inactive nodes.

Returns:
    The root of the estimated tree.
'''
def mlhd(data, alphabet, alpha, runs, complete=False, height_step=1):
    ml, mroot = None, None
    for r in range(runs):
        l, root = _mlhd(data, alphabet, alpha, complete, height_step)
        if ml is None or l > ml:
            ml, mroot = l, root
    return mroot

def _mlhd(data, alphabet, alpha, complete=False, height_step=1):
    root = create_tree(height_step, alphabet)
    initialise_counts(root, data, alphabet)
    activate(root, data, alphabet)
    l, increased = 1, True
    while increased:
        increased = False
        for a in permutation(root.attachment_count):
            v = attachment(root, a, complete)
            lr = birthratio(v, alpha, complete)
            if lr > 1:
                activate(v, data, alphabet, complete, height_step)
                increased = True
                l *= lr
                break
    return l, root