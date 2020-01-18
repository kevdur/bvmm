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
        i: the integer index assigned to this node's symbol.
        x: the symbol this node represents.
        parent: the first node along the path from this node to the root.
    '''
    def __init__(self, i, x, parent):
        self.index = i
        self.symbol = x
        self.counts = None
        self.parent = parent
        self.children = []
        self.node_count = 0 # no. of active nodes in the subtree rooted here.
        self.leaf_count = 0 # no. of active leaves.
        self.attachment_count = 0 # no. of valid (occurring), inactive nodes.
        self.sample_count = 0
        self.is_active = False
        self.checkpoints = [] # data indices at which this state appears.

class Options:
    '''
    A simple helper class used to ferry configuration options between functions.

    Args:
        full: whether or not the tree should be interpreted as a full tree, in
            which case leaves will be viewed as internal nodes, and their
            inactive children treated as leaves.
        fringe: if true, strict internal nodes (whose children are all active)
            will not have their sample counts updated.
        height_step: the number of levels that should be added each time the
            tree's limits are reached and it needs to be extended with new,
            inactive nodes.
        min_skip_prob: the base probability of proposing the 'identity' move,
            which does not alter the active tree at all. The effective
            probability will be greater whenever birth or death moves are
            impossible.
        kind: the data type, either 'sequence' or 'network'.
    '''
    def __init__(self, full=False, fringe=False, height_step=1,
            min_skip_prob=1/3, kind='sequence'):
        self.full = full
        self.fringe = fringe
        self.height_step = height_step
        self.min_skip_prob = min_skip_prob
        self.kind = kind

def create_tree(height, data, alphabet, kind='sequence'):
    '''
    Creates a full, inactive tree with initialised occurrence counts.

    Args:
        height: the depth to which the tree should be grown (a singleton tree
            has a height of zero).
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
        kind: the data type, either 'sequence' or 'network'.

    Returns:
        The root node of the tree.
    '''
    root = Node(-1, 'λ', None)
    _add_children(root, 1, height, alphabet)
    _initialise_counts(root, data, alphabet, kind)
    if root.counts is not None:
        root.attachment_count = 1
    return root

def _add_children(v, depth, max_depth, alphabet):
    if depth <= max_depth:
        v.children = [Node(i, x, v) for i, x in enumerate(alphabet)]
        for w in v.children:
            _add_children(w, depth+1, max_depth, alphabet)

def _initialise_counts(v, data, alphabet, kind):
    v.counts = np.zeros(len(alphabet))
    v.checkpoints = []
    if kind.lower() == 'sequence':
        count_func = _sequence_counts
    elif kind.lower() == 'network':
        count_func = _network_counts
    else:
        raise ValueError("Invalid data type specified. Valid options are "
                         "'sequence' and 'network'.")
    try:
        for i, array in enumerate(data):
            iter(array)
            if kind == 'network' and len(array) > 0:
                iter(array[0]) # the array should contain tuples.
            count_func(v, array, alphabet, i)
    except TypeError:
        count_func(v, data, alphabet, 0)

def _init_structs(v, alphabet, array_index):
    if v.counts is None:
        v.counts = np.zeros(len(alphabet))
    while len(v.checkpoints) <= array_index:
        v.checkpoints.append([])

def _sequence_counts(v, array, alphabet, array_index):
    # To avoid traversing the entire array, each node keeps track of the indices
    # at which its state (prefix string) appears -- specifically, the index of
    # the first character following the prefix string. A node's checkpoints are
    # necessarily a subset of those of its parent.
    m = len(path_to(v)) # length of the prefix.
    u = v.parent
    uchks = u.checkpoints[array_index] if u is not None else range(len(array))
    for j in uchks:
        i = j - m # first index of the prefix.
        if array[i] == v.index or v.parent is None:
            w = v
            while True: # update counts along the (extended) prefix's path.
                _init_structs(w, alphabet, array_index)
                w.counts[array[j]] += 1
                w.checkpoints[array_index].append(j)
                if i > 0 and w.children:
                    i -= 1
                    w = w.children[array[i]]
                else:
                    break

def _network_counts(v, array, alphabet, array_index):
    # Like we do for sequence data, we maintain checkpoints of where each state
    # appears so that we can avoid having to traversing the whole data array
    # each time we want to initialise a set of nodes' counts. Doing so for
    # network data is tougher however, because states don't appear as a simple
    # set of consecutive symbols -- the entries that make up a prefix path in
    # the network are unlikely to be consecutive, and a single entry can be a
    # part of multiple states that appear further on in the data. For example,
    # the entries (a, b), ..., (b, c), (c, d), ..., (b, c) include the state
    # 'ab' twice -- both times leading to the symbol 'c'. To keep track of all
    # of this, each checkpoint is a triple, of preceding index, final symbol,
    # and count. Preceding index refers to the index of the entry that would
    # need to be added if the state were to be extended; for example the
    # checkpoint for the state 'ab' with respect to the entries above (ignoring
    # the omitted entries) would be (0, 'c', 2), and the checkpoints for 'abc'
    # would be (1, 'd', 1). To make it easy to find preceding entries without
    # traversing the data in reverse, we also initialise a map that contains,
    # for each symbol, the indices of the entries in which it appears as a
    # destination (second element).
    dest_chks = _network_dests(v, array, alphabet, array_index)
    if v.parent is None:
        _init_structs(v, alphabet, array_index)
        chks = v.checkpoints[array_index]
        for k, (i, j) in enumerate(array):
            v.counts[j] += 1
            chks.append((k, j, 1))
        for w in v.children:
            _network_counts(w, array, alphabet, array_index)
    else:
        for k, j, c in v.parent.checkpoints[array_index]:
            if array[k][0] == v.index:
                w = v
                while True:
                    _init_structs(w, alphabet, array_index)
                    w.counts[j] += c
                    # Find the index that can be used to extend the state.
                    i = array[k][0]
                    if dest_chks[i] is None:
                        break
                    d = np.searchsorted(dest_chks[i], k) - 1 # last less than k.
                    if d < 0:
                        break
                    k = dest_chks[i][d]
                    w.checkpoints[array_index].append((k, j, c))
                    if not w.children:
                        break
                    w = w.children[array[k][0]]
        
def _network_dests(v, array, alphabet, array_index):
    root = _root(v)
    if not hasattr(root, '_dest_checkpoints'):
        root._dest_checkpoints = []
    while len(root._dest_checkpoints) <= array_index:
        root._dest_checkpoints.append(None)
    if root._dest_checkpoints[array_index] is None:
        # We store destination checkpoints as an array of indices for each
        # possible destination symbol.
        dest_chks = [None] * len(alphabet)
        for k, (i, j) in enumerate(array):
            if dest_chks[j] is None:
                dest_chks[j] = []
            dest_chks[j].append(k)
        root._dest_checkpoints[array_index] = dest_chks
    return root._dest_checkpoints[array_index]

def update_counts(v, nodes=0, leaves=0, attachments=0):
    '''
    Increases or decreases counts along the path from a node to the root.
    '''
    v.node_count += nodes
    v.leaf_count += leaves
    v.attachment_count += attachments
    if v.parent is not None:
        update_counts(v.parent, nodes, leaves, attachments)

def update_sample_counts(v, samples, opts):
    '''
    Increases (or decreases) the sample counts of the nodes in a tree.

    Args:
        v: the root of the subtree to be updated.
        samples: the amount that should be added to each sample count.
    '''
    if opts.full and v.parent is not None and not v.parent.is_active:
        return
    elif not opts.full and not v.is_active:
        return

    # A node is a fringe node if it is a leaf or if any of its valid children
    # are inactive.
    on_fringe = (not v.is_active) if opts.full else (v.node_count == 1)
    for w in v.children:
        if w.counts is not None:
            update_sample_counts(w, samples, opts)
            if not opts.full and not w.is_active:
                on_fringe = True
    if not opts.fringe or on_fringe:
        v.sample_count += samples

def path_to(v):
    '''
    Returns the path from the root to a given node as a list of symbols/indices.
    '''
    if v.parent is None:
        return []
    path = path_to(v.parent)
    path.append(v.index)
    return path

def _root(v):
    root = v
    while root.parent is not None:
        root = root.parent
    return root

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

def activate(v, data, alphabet, opts):
    '''
    Activates a node and, if necessary, initialises its children.

    If the `full` option is true, this function will initialise the descendants
    of `v` to a depth of two where necessary, because in the full case a child
    node will only be counted as a possible attachment if at least one of its
    children is valid as well.

    Args:
        v: the valid attachment node to be activated: `v` must be inactive, with
            an active parent, and have at least one positive count.
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: the set of characters that appear in the original data set.
    '''
    v.is_active = True
    v.node_count = 1
    v.leaf_count = 1
    v.attachment_count = 0
    if not opts.full:
        if not v.children:
            _add_children(v, 1, opts.height_step, alphabet)
            _initialise_counts(v, data, alphabet, opts.kind)
        for w in v.children:
            if w.counts is not None:
                w.attachment_count = 1
                v.attachment_count += 1
    else:
        # In the full case, a child node is a valid attachment if it has valid
        # children (with non-zero occurrence counts) of its own. If these
        # depth-two descendants have not yet been created, we know that v has
        # never been activated, and we can safely reinitialise its children
        # without affecting any sample counts.
        for w in v.children:
            if w.counts is not None and not w.children:
                v.children = []
                _add_children(v, 1, opts.height_step+1, alphabet)
                _initialise_counts(v, data, alphabet, opts.kind)
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