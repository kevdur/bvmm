#===============================================================================
# BVMM
# Input and Output Functions
#===============================================================================

def create_index(data, kind='sequence'):
    '''
    Converts a character-based data set to an integer-based one.

    Args:
        data: if `kind` is 'sequence', a list containing consecutive symbol
            strings, or a list of such lists. If `kind` is 'network', one or
            more lists containing pairs of symbols.
        kind: either 'sequence' or 'network'.

    Returns:
        The integer-based data set, along with a second list containing the
        symbol corresponding to each integer (accessible via `alphabet[i]`).
    '''
    def index_sequence(array):
        if isinstance(array, str):
            raise TypeError
        output = []
        for x in array:
            i = index.setdefault(x, len(index))
            output.append(i)
        return output

    def index_network(array):
        if array and isinstance(array[0], str):
            raise TypeError
        output = []
        for x, y in array:
            i = index.setdefault(x, len(index))
            j = index.setdefault(y, len(index))
            output.append((i, j))
        return output
    
    index = {}
    if kind.lower() == 'sequence':
        index_func = index_sequence
    elif kind.lower() == 'network':
        index_func = index_network
    else:
        raise ValueError("Invalid data type specified. Valid options are "
                         "'sequence' and 'network'.")
    try:
        return [index_func(array) for array in data], list(index.keys())
    except TypeError:
        return index_func(data), list(index.keys())

def apply_alphabet(data, alphabet, kind='sequence'):
    '''
    Converts an integer-based data set to a character-based one.

    Args:
        data: one or more lists, containing consecutive indices if `kind` is
            'sequence', or pairs of indices if `kind` is 'network'.
        alphabet: a list containing the symbols represented by the indices.
        kind: the data type, either 'sequence' or 'network'.
    '''
    def apply_sequence(array):
        return [alphabet[i] for i in array]
    def apply_network(array):
        return [(alphabet[i], alphabet[j]) for i, j in array]

    if kind.lower() == 'sequence':
        apply_func = apply_sequence
    elif kind.lower() == 'network':
        apply_func = apply_network
    else:
        raise ValueError("Invalid data type specified. Valid options are "
                         "'sequence' and 'network'.")
    try:
        return [apply_func(array) for array in data]
    except TypeError:
        return apply_func(data)

def print_tree(v, alphabet, complete=False, min_samples=1e-16, max_counts=None,
        verbose=False, prefix=''):
    '''
    Prints details of a tree's nodes, in depth-first order.

    Nodes are printed if they are active (or in the case of complete trees if
    they have an active parent) or if their sample count is greater than or
    equal to the given minimum. Nodes with no associated symbol counts are
    hidden by default.

    Args:
        alphabet: a list containing the symbols represented by the indices.
        complete: whether or not the tree should be interpreted as a complete
            tree, in which case leaves will be viewed as internal nodes, and
            their inactive children treated as leaves.
        min_samples: nodes with fewer than `min_samples` samples will not be
            printed.
        max_counts: if specified, the number of symbol counts printed for each
            node will be limited.
        verbose: if true, nodes with no associated symbol counts will also be
            printed, along with debugging information for each node.
        prefix: a prefix string to attach to all of the printed nodes (used
            internally for recursive calls).
    '''
    valid = _valid(v, complete, min_samples) 
    if not valid or v.counts is None and not verbose:
        return

    prefix += str(v.symbol)
    smpl = _num_str(v.sample_count, 3, 5)
    if max_counts is None:
        max_counts = len(alphabet)
    if v.counts is None:
        cnts = 'None'
    else:
        cnts = sorted(zip(alphabet, v.counts), key=lambda z: z[1], reverse=True)
        cnts = cnts[:max_counts]
        cnts = ', '.join(x + ': ' + _num_str(c) for x, c in cnts if c > 0)

    if verbose:
        print('{:5} {} [{}]'.format(prefix + ':', smpl, cnts))
        print(' '*12 + '({}active, {} nds, {} lvs, {} atts)'.format(
            '' if v.is_active else 'in',
            v.node_count, v.leaf_count, v.attachment_count
        ))
    else:
        print('{:5} {} [{}]'.format(prefix + ':', smpl, cnts))
    for w in v.children:
        print_tree(w, alphabet, complete, min_samples, max_counts, verbose,
                   prefix)

def _num_str(x, decimal=2, padding=0):
    if x == int(x):
        return '{{:{}.0f}}'.format(padding).format(x)
    else:
        return '{{:{}.{}f}}'.format(padding, decimal).format(x)

def write_tree(v, alphabet, filename, complete=False, min_samples=1e-16,
        prefix=''):
    '''
    Writes a simple representation of a tree to a .net (modified Pajek) file.

    Nodes are printed if they are active (or in the case of complete trees if
    they have an active parent) or if their sample count is greater than or
    equal to the given minimum. Nodes with no associated symbol counts are
    ignored.

    Args:
        alphabet: a list containing the symbols represented by the indices.
        filename: the full path of the output file.
        complete: whether or not the tree should be interpreted as a complete
            tree, in which case leaves will be viewed as internal nodes, and
            their inactive children treated as leaves.
        min_samples: nodes with fewer than `min_samples` samples will not be
            printed.
        prefix: a prefix string to attach to all of the printed nodes.
    '''
    def append_node(v, nodes):
        nodes.append(v)
        return nodes
    def write_node(v, prefix):
        prefix += str(v.symbol)
        f.write('{} "{}" {}\n'.format(nodes[v], prefix, v.sample_count))
        return prefix
    def write_edges(v):
        if v.parent is not None:
            f.write('{} {}\n'.format(nodes[v.parent], nodes[v]))

    nodes = _visit_valid(append_node, v, complete, min_samples, [])
    nodes = {v: i+1 for i, v in enumerate(nodes)} # number the nodes.
    if len(nodes) == 0:
        return
    with open(filename, 'w') as f:
        f.write('*Vertices {}\n'.format(len(nodes)))
        _visit_valid(write_node, v, complete, min_samples, prefix)
        f.write('*Edges {}\n'.format(len(nodes)-1))
        _visit_valid(write_edges, v, complete, min_samples)

def _valid(v, complete=False, min_samples=1e-16):
    if v.sample_count < min_samples:
        if complete and v.parent is not None and not v.parent.is_active:
            return False
        elif not complete and not v.is_active:
            return False
    return True

def _visit_valid(f, v, complete=False, min_samples=1e-16, args=None):
    if not _valid(v, complete, min_samples):
        return
    if args is not None:
        args = f(v, args)
    else:
        f(v)
    for w in v.children:
        _visit_valid(f, w, complete, min_samples, args)
    return args