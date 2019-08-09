#===============================================================================
# BVMM
# Tree Manipulation Functions
#===============================================================================

from numpy import zeros

'''
Represents a node in a tree, and provides access to the subtree rooted there.

Each node represents a symbol in the alphabet derived from a set of input data,
and every path from the root of a tree to one of its nodes (including the root
itself) corresponds to a state of the implied Markov model. Instead of growing
or pruning the tree to alter the model it represents, nodes on the fringes of
the tree are simply activated or deactivated.

Each node maintains a count of the number of times the corresponding state
occurs in the data, along with counts for the symbols that follow those
occurrences, allowing probabilities to be computed using the tree directly.
Nodes also maintain certain statistics about their subtrees, facilitating the
traversal, growth, and pruning of the tree.

Args:
    i: the symbol this node represents.
    parent: the first node along the path from this node to the root.
'''
class Node:
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

'''
Creates a complete, inactive tree with uninitialised occurrence counts.

Args:
    height: the depth to which the tree should be grown (a singleton tree has a
        height of zero).
    alphabet: the set of characters that appear in the original data set.

Returns:
    The root node of the tree.
'''
def create_tree(height, alphabet):
    root = Node('λ', None)
    _add_children(root, 1, height, alphabet)
    return root

def _add_children(v, depth, max_depth, alphabet):
    if depth <= max_depth:
        v.children = [Node(x, v) for x in alphabet]
        for w in v.children:
            _add_children(w, depth+1, max_depth, alphabet)

'''
Initialises the occurrence counts for each node of a given subtree.

Args:
    root: the root of the subtree to be initialised.
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
'''
def initialise_counts(root, data, alphabet):
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
        v.counts = zeros(len(alphabet))
    v.counts[i] += 1
    if n > 0 and v.children:
        _increment_counts(v.children[array[n-1]], i, n-1, array, alphabet)

'''
Increases or decreases auxiliary counts along the path from a node to the root.
'''
def update_counts(v, nodes=0, leaves=0, attachments=0):
    v.node_count += nodes
    v.leaf_count += leaves
    v.attachment_count += attachments
    if v.parent is not None:
        update_counts(v.parent, nodes, leaves, attachments)

'''
Increases, decreases, or scales the sample counts of the nodes in a tree.

Args:
    v: the root of the subtree to be updated.
    samples: the amount by which each sample count should be incremented. If
        not `None`, `scale` will be ignored.
    scale: the value with which each sample count should be scaled. Ignored if
        `samples` is not `None`.
    fringe: if true, strict internal nodes (whose children are all active) will
        not have their sample counts updated.
    complete: whether or not the tree should be interpreted as a complete tree,
        in which case leaves will be viewed as internal nodes, and their
        inactive children treated as leaves.
'''
def update_sample_counts(v, samples=1, scale=1, fringe=False, complete=False):
    if complete and v.parent is not None and not v.parent.is_active:
        return
    elif not complete and not v.is_active:
        return

    # A node is a fringe node if it is a leaf or if any of its valid children
    # are inactive.
    on_fringe = (not v.is_active) if complete else (v.node_count == 1)
    for w in v.children:
        if w.counts is not None:
            update_sample_counts(w, samples, scale, fringe, complete)
            if not complete and not w.is_active:
                on_fringe = True
    if not fringe or on_fringe:
        if samples is not None:
            v.sample_count += samples
        else:
            v.sample_count *= scale

'''
Returns the path from the root to a given node as a list of symbols/indices.
'''
def path_to(v):
    if v.parent is None:
        return []
    path = path_to(v.parent)
    path.append(v.symbol)
    return path

'''
Returns leaf `l` of a given subtree, indexed in depth-first order.
'''
def leaf(v, l):
    if v.node_count == 1:
        return v
    for w in v.children:
        if l < w.leaf_count:
            return leaf(w, l)
        l -= w.leaf_count

'''
Returns attachment `a` of a given subtree, indexed in depth-first order.
'''
def attachment(v, a):
    if v.node_count == 0:
        return v
    for w in v.children:
        if a < w.attachment_count:
            return attachment(w, a)
        a -= w.attachment_count

'''
Activates a node and, if necessary, initialises its children.

Args:
    v: the valid attachment node to be activated: `v` must be inactive, with an
        active parent, and have at least one positive count.
    data: a list of integer indices, or an iterable set of such lists.
    alphabet: the set of characters that appear in the original data set.
    height_step: the number of levels that should be added each time the tree's
        limits are reached and it needs to be extended with new, inactive nodes.
'''
def activate(v, data, alphabet, height_step=1):
    v.is_active = True
    v.node_count = 1
    v.leaf_count = 1
    v.attachment_count = 0
    if not v.children:
        _add_children(v, 1, height_step, alphabet)
        v.counts = zeros(len(alphabet))
        initialise_counts(v, data, alphabet)
    for w in v.children:
        if w.counts is not None:
            w.attachment_count = 1
            v.attachment_count += 1
    if v.parent is not None:
        leaves = 1 if v.parent.node_count > 1 else 0
        update_counts(v.parent, nodes=1, leaves=leaves,
                      attachments=v.attachment_count-1)

'''
Deactivates a node.

Args:
    v: the node to be deactivated: `v` must be an active leaf (all of its
        children should be inactive).
'''
def deactivate(v):
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

# '''
# Activates all of a node's children.

# This function should not be used in combination with the singular `activate`,
# since it assumes that all previous activations have been done in a complete way.

# Args:
#     u: the leaf whose children are to be activated: `u` must be active and have
#         at least one valid attachment node among its children.
#     data: a list of integer indices, or an iterable set of such lists.
#     alphabet: the set of characters that appear in the original data set.
#     height_step: the number of levels that should be added each time the tree's
#         limits are reached and it needs to be extended with new, inactive nodes.
# '''
# def activate_children(u, data, alphabet, height_step=1):
#     # This function is more efficient than calling 'activate' on the child nodes
#     # individually because it only calls 'initialise_counts' and 'update_counts'
#     # once.
#     u.leaf_count = 0
#     u.attachment_count = 0
#     for v in u.children:
#         if v.counts is not None and not v.children:      # implies none of u's
#             _add_children(u, 1, height_step+1, alphabet) # children have been
#             u.counts = zeros(len(alphabet))              # activated yet.
#             initialise_counts(u, data, alphabet)
#             break
#     for v in u.children:
#         if v.counts is None:
#             continue
#         v.is_active = True
#         v.node_count = 1
#         v.leaf_count = 1
#         v.attachment_count = 0
#         u.node_count += 1
#         u.leaf_count += 1
#         for w in v.children:
#             if w.counts is not None:
#                 w.attachment_count = 1
#                 v.attachment_count += 1
#                 u.attachment_count += 1
#     update_counts(u.parent, nodes=u.leaf_count, leaves=u.leaf_count-1,
#                   attachments=u.attachment_count-u.leaf_count)

# '''
# Deactivates all of a node's children.

# This function should not be used in combination with the singular `deactivate`.

# Args:
#     v: the node whose children are to be deactivated: `u` must be active, and
#         all of its (valid) children must be active leaf nodes (their children
#         should be inactive).
# '''
# def deactivate_children(u):
#     u.node_count = 1
#     for v in u.children:
#         if v.counts is None:
#             continue
#         v.is_active = False
#         v.node_count = 0
#         v.leaf_count = 0
#         if v.attachment_count > 0:
#             for w in v.children:
#                 w.attachment_count = 0
#         v.attachment_count = 1
#     update_counts(u.parent, nodes=-u.leaf_count, leaves=1-u.leaf_count,
#                   attachments=u.leaf_count-u.attachment_count)
#     u.attachment_count = u.leaf_count
#     u.leaf_count = 1