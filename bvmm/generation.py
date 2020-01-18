#===============================================================================
# BVMM
# Tree and Data Generation Functions
#===============================================================================

import numpy as np
from . import tree
from . import likelihood

def rand_tree(n, alphabet, alpha=None):
    '''
    Returns a random tree whose counts are random probability vectors.

    Args:
        n: the number of active nodes in the tree.
        alphabet: the set of characters that should be associated with the
            tree's nodes.
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet distribution from which the nodes' counts are sampled. It
            should have the same length as the alphabet array.
    '''
    alpha = likelihood._verify_alpha(alpha, alphabet)
    opts = tree.Options(False, height_step=1)
    root = tree.create_tree(0, [], alphabet)
    _rand_counts(root, alpha)
    for i in range(n):
        v = tree.attachment(root, np.random.randint(0, root.attachment_count))
        tree.activate(v, [], alphabet, opts)
        _rand_counts(v, alpha)
    return root

def _rand_counts(v, alpha):
    '''
    Assigns counts to a subtree, and marks nodes as attachments where necessary.
    '''
    if v.counts is None:
        tree.update_counts(v, attachments=1)
    v.counts = np.random.dirichlet(alpha)
    for w in v.children:
        _rand_counts(w, alpha)

def rand_data(root, n):
    '''
    Returns an array of data generated (recursively) from a given tree.

    Args:
        root: the root of the tree used for data generation.
        n: the desired length of the data array.
    '''
    indices = list(range(len(root.counts)))
    data = []
    for i in range(n):
        v = _node_from(root, data)
        data.append(np.random.choice(indices, p=v.counts/v.counts.sum()))
    return data

def _node_from(v, prefix):
    '''
    Returns the node reached by traversing the given list of indices in reverse.
    '''
    i = 1
    while len(prefix) >= i and v.children[prefix[-i]].is_active:
        v = v.children[prefix[-i]]
        i += 1
    return v