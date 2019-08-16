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