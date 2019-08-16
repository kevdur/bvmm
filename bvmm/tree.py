#===============================================================================
# BVMM
# Tree Manipulation Functions
#===============================================================================

import collections
import numpy as np

class Node:
    '''
    Represents a node in a tree and provides access to the subtree rooted there.

    Each node represents a symbol in the alphabet derived from a set of input
    data, and every path from the root of a tree to one of its nodes (including
    the root itself) corresponds to a state of the implied Markov model. Instead
    of growing or pruning the tree to alter the model it represents, nodes on
    the fringes of the tree are simply activated or deactivated.

    Each node maintains a count of the number of times the corresponding state
    occurs in the data, along with counts for the symbols that follow those
    occurrences, allowing probabilities to be computed using the tree directly.
    Nodes also maintain certain statistics about their subtrees, facilitating
    the traversal, growth, and pruning of the tree.

    Args:
        i: the symbol this node represents. parent: the first node along the
            path from this node to the root.
    '''
    def __init__(self, i, parent):
        self.symbol = i
        self.counts = None
        self.parent = parent
        self.children = []
        self.node_count = 0 # no. of active nodes in the subtree rooted here.
        self.leaf_count = 0 # no. of active leaves.
        self.attachment_count = 0 # no. of valid (occurring), inactive nodes.
        self.sample_count = 0
        self.is_active = False

class Options:
    '''
    A simple helper class used to ferry configuration options between functions.

    Args:
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
    '''
    def __init__(self, complete=False, fringe=False, height_step=1,
            min_skip_prob=1/3):
        self.complete = complete
        self.fringe = fringe
        self.height_step = height_step
        self.min_skip_prob = min_skip_prob

def create_tree(height, alphabet):
    '''
    Creates a complete, inactive tree with uninitialised occurrence counts.

    Args:
        height: the depth to which the tree should be grown (a singleton tree
            has a height of zero).
        alphabet: the set of characters that appear in the original data set.

    Returns:
        The root node of the tree.
    '''
    root = Node('λ', None)
    _add_children(root, 1, height, alphabet)
    return root

def _add_children(v, depth, max_depth, alphabet):
    if depth <= max_depth:
        v.children = [Node(x, v) for x in alphabet]
        for w in v.children:
            _add_children(w, depth+1, max_depth, alphabet)

def initialise_counts(root, data, alphabet):
    '''
    Initialises the occurrence counts for each node of a given subtree.

    Args:
        root: the root of the subtree to be initialised.
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
    '''
    # This function traverses an array of data in reverse, at each datum
    # incrementing counts along a path from the root of the subtree to one of
    # its leaves, according to the datum's prefix.
    def array_counts(array):
        for n in range(len(array)-1, len(state)-1, -1):
            if array[n-len(state):n] == state:
                _increment_counts(root, array[n], n-len(state), array, alphabet)
    state = path_to(root)[::-1]
    try:
        map(array_counts, data)
    except TypeError:
        array_counts(data)

def _increment_counts(v, i, n, array, alphabet):
    if v.counts is None:
        v.counts = np.zeros(len(alphabet))
    v.counts[i] += 1
    if n > 0 and v.children:
        _increment_counts(v.children[array[n-1]], i, n-1, array, alphabet)

def update_counts(v, nodes=0, leaves=0, attachments=0):
    '''
    Increases or decreases counts along the path from a node to the root.
    '''
    v.node_count += nodes
    v.leaf_count += leaves
    v.attachment_count += attachments
    if v.parent is not None:
        update_counts(v.parent, nodes, leaves, attachments)

def update_sample_counts(v, samples=1, scale=1, opts=Options()):
    '''
    Increases, decreases, or scales the sample counts of the nodes in a tree.

    Args:
        v: the root of the subtree to be updated.
        samples: the amount by which each sample count should be incremented. If
            not `None`, `scale` will be ignored.
        scale: the value with which each sample count should be scaled. Ignored
            if `samples` is not `None`.
    '''
    if opts.complete and v.parent is not None and not v.parent.is_active:
        return
    elif not opts.complete and not v.is_active:
        return

    # A node is a fringe node if it is a leaf or if any of its valid children
    # are inactive.
    on_fringe = (not v.is_active) if opts.complete else (v.node_count == 1)
    for w in v.children:
        if w.counts is not None:
            update_sample_counts(w, samples, scale, opts)
            if not opts.complete and not w.is_active:
                on_fringe = True
    if not opts.fringe or on_fringe:
        if samples is not None:
            v.sample_count += samples
        else:
            v.sample_count *= scale

def path_to(v):
    '''
    Returns the path from the root to a given node as a list of symbols/indices.
    '''
    if v.parent is None:
        return []
    path = path_to(v.parent)
    path.append(v.symbol)
    return path

def leaf(v, l):
    '''
    Returns leaf `l` of a given subtree, indexed in depth-first order.
    '''
    if v.node_count == 1:
        return v
    for w in v.children:
        if l < w.leaf_count:
            return leaf(w, l)
        l -= w.leaf_count

def attachment(v, a):
    '''
    Returns attachment `a` of a given subtree, indexed in depth-first order.
    '''
    if v.node_count == 0:
        return v
    for w in v.children:
        if a < w.attachment_count:
            return attachment(w, a)
        a -= w.attachment_count

def activate(v, data, alphabet, opts=Options()):
    '''
    Activates a node and, if necessary, initialises its children.

    If the `complete` option is true, this function will initialise the
    descendants of `v` to a depth of two where necessary, because in the
    complete case a child node will only be counted as a possible attachment if
    at least one of its children is valid as well.

    Args: v: the valid attachment node to be activated: `v` must be inactive,
        with an active parent, and have at least one positive count. data: a
        list of integer indices, or an iterable set of such lists. alphabet: the
        set of characters that appear in the original data set.
    '''
    v.is_active = True
    v.node_count = 1
    v.leaf_count = 1
    v.attachment_count = 0
    if not opts.complete:
        if not v.children:
            _add_children(v, 1, opts.height_step, alphabet)
            v.counts = np.zeros(len(alphabet))
            initialise_counts(v, data, alphabet)
        for w in v.children:
            if w.counts is not None:
                w.attachment_count = 1
                v.attachment_count += 1
    else:
        # In the complete case, a child node is a valid attachment if it has
        # valid children (with non-zero occurrence counts) of its own. If these
        # depth-two descendants have not yet been created, we know that v has
        # never been activated, and we can safely reinitialise its children
        # without affecting any sample counts.
        for w in v.children:
            if w.counts is not None and not w.children:
                v.children = []
                _add_children(v, 1, opts.height_step+1, alphabet)
                v.counts = np.zeros(len(alphabet))
                initialise_counts(v, data, alphabet)
                break
        for w in v.children:
            if w.counts is None:
                continue
            if any(x.counts is not None for x in w.children):
                w.attachment_count = 1
                v.attachment_count += 1
    if v.parent is not None:
        leaves = 1 if v.parent.node_count > 1 else 0
        update_counts(v.parent, nodes=1, leaves=leaves,
                      attachments=v.attachment_count-1)

def deactivate(v):
    '''
    Deactivates a node.

    Args:
        v: the node to be deactivated: `v` must be an active leaf (all of its
            children should be inactive).
    '''
    v.is_active = False
    v.node_count = 0
    v.leaf_count = 0
    if v.parent is not None:
        leaves = -1 if v.parent.node_count > 2 else 0
        update_counts(v.parent, nodes=-1, leaves=leaves,
                      attachments=1-v.attachment_count)
    if v.attachment_count > 0:
        for w in v.children:
            w.attachment_count = 0
    v.attachment_count = 1