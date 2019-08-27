#===============================================================================
# BVMM
# Input and Output Functions
#===============================================================================

def create_alphabet(data):
    '''
    Maps one or more character strings to arrays of zero-based integers.

    Args:
        data: a string, or an iterable set of strings.

    Returns:
        The resulting list(s) of indices, along with a second list containing
        the symbols corresponding to these integers (accessible via
        `alphabet[i]`).
    '''
    output, index = [], {}
    if isinstance(data, str):
        for x in data:
            i = index.setdefault(x, len(index))
            output.append(i)
    else:
        for string in data:
            array = []
            for x in string:
                i = index.setdefault(x, len(index))
                array.append(i)
            output.append(array)
    return output, list(index.keys)

def apply_alphabet(data, alphabet):
    '''
    Maps data in the form of integer arrays to character strings.

    Args:
        data: a list of integer indices, or an iterable set of such lists.
        alphabet: a list containing the symbols represented by the indices.

    Returns:
        A string or a list of strings, depending on the form of `data`.
    '''
    try:
        return [''.join(alphabet[i] for i in array) for array in data]
    except TypeError:
        return ''.join(alphabet[i] for i in data)

def print_tree(v, complete=False, min_samples=1e-16, verbose=False, prefix=''):
    '''
    Prints details of a tree's nodes, in depth-first order.

    By default, nodes will be printed if they are active (or in the case of
    complete trees if they have an active parent) or if their sample count
    is greater than or equal to the given minimum.

    Args:
        complete: whether or not the tree should be interpreted as a complete
            tree, in which case leaves will be viewed as internal nodes, and
            their inactive children treated as leaves.
        min_samples: nodes with fewer than `min_samples` samples will not be
            printed.
        verbose: if true, extra information will be printed for each node.
        prefix: a prefix string to attach to all of the printed nodes (used
            internally for recursive calls).
    '''
    if v.sample_count < min_samples:
        if complete and v.parent is not None and not v.parent.is_active:
            return
        elif not complete and not v.is_active:
            return

    smpl = '{:5}' if v.sample_count == int(v.sample_count) else '{:5.3f}'
    if verbose:
        print('{:5} {} {} ({}active, {} nds, {} lvs, {} atts)'.format(
            prefix + str(v.symbol) + ':',
            smpl.format(v.sample_count),
            v.counts if v.counts is not None else '[None]',
            '' if v.is_active else 'in',
            v.node_count, v.leaf_count, v.attachment_count
        ))
    else:
        print('{:5} {} {}'.format(
            prefix + str(v.symbol) + ':',
            smpl.format(v.sample_count),
            v.counts if v.counts is not None else '[None]'
        ))
    for w in v.children:
        print_tree(w, complete, min_samples, verbose, prefix + str(v.symbol))