#===============================================================================
# Bayesian Variable-order Markov Models
# Kevin Durant
# June 2019
#===============================================================================

# Notes
# 1. There must be a 'remain' probability so that ergodicity is satisfied.
# 2. Binary data should generally be handled using complete trees.

import numpy as np
from itertools import product
from numpy.random import choice, dirichlet, permutation, rand, randint
from scipy.special import gammaln

# Main method ==================================================================

def bvmm(data, symbols, alpha, samples):
    root = create_tree(1, symbols)
    initialise_counts(root, data, symbols)
    activate(root, data, symbols)
    ss, bs, bss, ds, dss = 0, 0, 0, 0, 0
    for i in range(samples):
        birth_prob, death_prob = move_probabilities(root)
        m = rand()
        if m < birth_prob:
            bs += 1
            if birth(root, data, symbols, alpha):
                bss += 1
        elif m < birth_prob + death_prob:
            ds += 1
            if death(root, data, symbols, alpha):
                dss += 1
        else:
            ss += 1
        increment_sample_counts(root)
    print('Skips:', ss)
    print('Births: {}/{}'.format(bss, bs))
    print('Deaths: {}/{}'.format(dss, ds))
    return root

# Pre-processing ===============================================================

def create_index(string):
    data, index = [], {}
    for s in string:
        i = index.setdefault(s, len(index))
        data.append(i)
    return data, index

def apply_index(data, index):
    alphabet = list(index.keys())
    return ''.join(alphabet[i] for i in data)

# Tree construction ============================================================

class Node:
    def __init__(self, i, parent):
        self.symbol = i # an integer.
        self.counts = None
        self.parent = parent
        self.children = []
        self.node_count = 0 # no. of active nodes in the subtree rooted here.
        self.leaf_count = 0 # no. of active leaves.
        self.attachment_count = 0 # no. of valid, inactive nodes.
        self.sample_count = 0
        self.is_active = False

def create_tree(height, symbols):
    root = Node('λ', None)
    add_children(root, 1, height, symbols)
    return root

def add_children(v, depth, max_depth, symbols):
    if depth <= max_depth:
        v.children = [Node(i, v) for i in symbols]
        for w in v.children:
            add_children(w, depth+1, max_depth, symbols)

def path_to(v):
    if v.parent is None:
        return []
    path = path_to(v.parent)
    path.append(v.symbol)
    return path

def initialise_counts(v, data, symbols):
    state = path_to(v)[::-1]
    for n in range(len(data)-1, len(state)-1, -1):
        if data[n-len(state):n] == state:
            increment_counts(v, data[n], n-len(state), data, symbols)

def increment_counts(v, i, n, data, symbols):
    if v.counts is None:
        v.counts = [0] * len(symbols)
    v.counts[i] += 1
    if n > 0 and v.children:
        increment_counts(v.children[data[n-1]], i, n-1, data, symbols)

def update_counts(v, nodes=0, leaves=0, attachments=0):
    v.node_count += nodes
    v.leaf_count += leaves
    v.attachment_count += attachments
    if v.parent is not None:
        update_counts(v.parent, nodes, leaves, attachments)

# Sampling =====================================================================

# Returns proposal probabilities for the birth and death moves, in that order.
def move_probabilities(root, min_skip_prob=1/3):
    move_prob = 1-min_skip_prob
    if root.attachment_count == 0 and root.node_count == 1:
        return 0, 0
    if root.attachment_count == 0:
        return 0, move_prob
    if root.node_count == 1:
        return move_prob, 0
    return .5*move_prob, .5*move_prob

def rand_leaf(v):
    if v.node_count == 1:
        return v
    l = randint(0, v.leaf_count)
    for w in v.children:
        if l < w.leaf_count:
            return rand_leaf(w)
        l -= w.leaf_count

def rand_attachment(v):
    if v.node_count == 0:
        return v
    a = randint(0, v.attachment_count)
    for w in v.children:
        if a < w.attachment_count:
            return rand_attachment(w)
        a -= w.attachment_count

# Assume that v is a valid attachment, i.e. is inactive, has an active parent,
# and has at least one positive count.
def activate(v, data, symbols):
    v.is_active = True
    v.node_count = 1
    v.leaf_count = 1
    v.attachment_count = 0
    if not v.children:
        add_children(v, 1, 1, symbols)
        v.counts = [0] * len(symbols)
        initialise_counts(v, data, symbols)
    for w in v.children:
        if w.counts is not None:
            w.attachment_count = 1
            v.attachment_count += 1
    if v.parent is not None:
        leaves = 1 if v.parent.node_count > 1 else 0
        update_counts(v.parent, nodes=1, leaves=leaves,
                      attachments=v.attachment_count-1)

# Assume that v is a leaf, i.e. is active but has no active children.
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

# Assume v is a leaf.
def lhdratio(v, alpha):
    l = 0
    alt_counts = np.array(v.parent.counts)
    for w in v.parent.children:
        if w.node_count > 0 and w != v:
            alt_counts -= w.counts
    for uc, vc, a in zip(alt_counts, v.counts, alpha):
        l += gammaln(uc+a) - gammaln(uc-vc+a) + gammaln(a) - gammaln(vc+a)
    usm, vsm, asm = sum(alt_counts), sum(v.counts), sum(alpha)
    l += gammaln(usm-vsm+asm) + gammaln(vsm+asm)
    l -= gammaln(usm+asm) + gammaln(asm)
    return np.exp(l)

def birth(root, data, symbols, alpha):
    v = rand_attachment(root)
    l = 1/lhdratio(v, alpha)
    birth_prob = move_probabilities(root)[0]/root.attachment_count
    activate(v, data, symbols)
    death_prob = move_probabilities(root)[1]/root.leaf_count
    if rand() > l*death_prob/birth_prob:
        deactivate(v)
        return False
    return True

def death(root, data, symbols, alpha):
    v = rand_leaf(root)
    l = lhdratio(v, alpha)
    death_prob = move_probabilities(root)[1]/root.leaf_count
    deactivate(v)
    birth_prob = move_probabilities(root)[0]/root.attachment_count
    if rand() > l*birth_prob/death_prob:
        activate(v, data, symbols)
        return False
    return True

def increment_sample_counts(v, samples=1):
    if v.node_count > 0:
        v.sample_count += samples
    for w in v.children:
        increment_sample_counts(w, samples)

# Optimisation =================================================================

def lhd(root, data, symbols, alpha):
    l, vs = 1, []
    while root.node_count > 1:
        v = leaf(root, 0)
        l *= lhdratio(v, alpha)
        deactivate(v)
        vs.append(v)
    for v in reversed(vs):
        activate(v, data, symbols)
    return 1/l

def mvmm(data, symbols, alpha, runs):
    ml, mroot = None, None
    for i in range(runs):
        l, root = mlhd(data, symbols, alpha)
        if ml is None or l > ml:
            ml, mroot = l, root
    return mroot

def mlhd(data, symbols, alpha):
    l = 1
    root = create_tree(1, symbols)
    initialise_counts(root, data, symbols)
    activate(root, data, symbols)
    increased = True
    while increased:
        increased = False
        for a in permutation(root.attachment_count):
            v = attachment(root, a)
            activate(v, data, symbols)
            lr = 1/lhdratio(v, alpha)
            if lr <= 1:
                deactivate(v)
                continue
            l *= lr
            increased = True
            break
    return l, root

def leaf(v, l):
    if v.node_count == 1:
        return v
    for w in v.children:
        if l < w.leaf_count:
            return leaf(w, l)
        l -= w.leaf_count

def attachment(v, a):
    if v.node_count == 0:
        return v
    for w in v.children:
        if a < w.attachment_count:
            return attachment(w, a)
        a -= w.attachment_count

# Assume v is inactive.
def subtrees(v, i, data, symbols):
    if i == 0 and v.parent is not None:
        yield
        if v.counts is not None:
            activate(v, data, symbols)
    if v.is_active:
        if i == len(symbols) - 1:
            for s in subtrees(v.children[i], 0, data, symbols):
                yield
        else:
            for s in subtrees(v.children[i], 0, data, symbols):
                for t in subtrees(v, i+1, data, symbols):
                    yield
        if i == 0 and v.parent is not None:
            deactivate(v)

def lhds(data, symbols, alpha):
    root = create_tree(1, symbols)
    initialise_counts(root, data, symbols)
    activate(root, data, symbols)
    for s in subtrees(root.children[0], 0, data, symbols):
        for t in subtrees(root, 1, data, symbols):
            increment_sample_counts(root, lhd(root, data, symbols, alpha))
    scale_sample_counts(root)
    return root

def scale_sample_counts(v, f=None):
    if f is None:
        f = 1/v.sample_count if v.sample_count > 0 else 1
    v.sample_count *= f
    for w in v.children:
        scale_sample_counts(w, f)

# Data generation ==============================================================

def rand_tree(n, symbols, alpha):
    root = create_tree(0, symbols)
    rand_counts(root, alpha)
    for i in range(n):
        v = rand_attachment(root)
        activate(v, [], symbols)
        rand_counts(v, alpha)
    return root

def rand_counts(v, alpha):
    if v.counts is None and v.parent is not None and v.parent.is_active:
        update_counts(v, attachments=1)
    v.counts = dirichlet(alpha)
    for w in v.children:
        rand_counts(w, alpha)

def rand_data(root, data, n, symbols):
    if n == 0:
        return data
    v = node_from(root, data)
    data.append(choice(symbols, p=v.counts))
    return rand_data(root, data, n-1, symbols)

# Assume that v is active.
def node_from(v, prefix):
    if len(prefix) == 0 or not v.children[prefix[-1]].is_active:
        return v
    return node_from(v.children[prefix[-1]], prefix[:-1])

# Output =======================================================================

def print_tree(v, inactive=True, min_samples=0, prefix=''):
    if (v.is_active or inactive) and v.sample_count >= min_samples:
        smpl = '{}' if v.sample_count == int(v.sample_count) else '{:.3f}'
        print(prefix + str(v.symbol),
              '({}active,'.format('' if v.is_active else 'in'),
              '{} nds,'.format(v.node_count),
              '{} lvs,'.format(v.leaf_count),
              '{} atts,'.format(v.attachment_count),
              (smpl + ' smpls)').format(v.sample_count),
              v.counts)
        for w in v.children:
            print_tree(w, inactive, min_samples, prefix + str(v.symbol))

symbols = range(2)
alpha = [1, 1]
root = rand_tree(3, symbols, alpha)
print_tree(root)
data = rand_data(root, [], 4, symbols)
print(data)
mroot = mvmm(data, symbols, alpha, 10)

data = [0, 1, 0, 1]
data = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
data = [1, 2, 0, 2]
data = [0, 1, 2, 1, 0, 1, 2, 0, 2, 0, 1, 2]

root = create_tree(1, symbols)
initialise_counts(root, data, symbols)
activate(root, data, symbols)
for s in subtrees(root, 0, data, symbols):
    print_tree(root)
    print(lhd(root, data, symbols, alpha))
    print()

lroot = lhds(data, symbols, alpha)
print_tree(lroot, min_samples=1e-16)

broot = bvmm(data, symbols, alpha, 100000)
scale_sample_counts(broot)
print_tree(broot, min_samples=1e-16)