#===============================================================================
# BVMM
# Tree and Data Generation Functions
#===============================================================================

import numpy as np
from . import tree

def rand_tree(n, alphabet, alpha):
    '''
    Returns a random tree whose counts are random probability vectors.

    Args:
        n: the number of active nodes in the tree.
        alphabet: the set of characters that should be associated with the
            tree's nodes.
        alpha: the 'concentration' vector that is used to parameterise the
            Dirichlet distribution from which the nodes' counts are sampled.
    '''
    opts = tree.Options(False, height_step=1)
    root = tree.create_tree(0, alphabet)
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

def rand_data(root, n, alphabet, data=[]):
    '''
    Returns an array of data generated (recursively) from a given tree.

    Args:
        root: the root of the tree used for data generation.
        n: the desired length of the data array.
        alphabet: the set of characters that should appear in the data set.
        data: the array of data generated so far.
    '''
    if n == 0:
        return data
    v = _node_from(root, data)
    data.append(np.random.choice(alphabet, p=v.counts/v.counts.sum()))
    return rand_data(root, n-1, alphabet, data)

def _node_from(v, prefix):
    '''
    Returns the node reached by traversing the given list of indices in reverse.
    '''
    if len(prefix) == 0 or not v.children[prefix[-1]].is_active:
        return v
    return _node_from(v.children[prefix[-1]], prefix[:-1])