#%%
import bvmm
import pickle

#%%
def preprocess(infilename, outfilename, allow_compound=True):
    with open(infilename, 'r') as f, open(outfilename, 'w') as g:
        text = []
        v, w = '', ''
        for x in f.read():
            if w.isspace() or w.isalnum():
                text.append(w.lower())
            elif v.isalnum() and x.isalnum():
                if allow_compound:
                    text.append(w)
            else:
                text.append(' ')
            v, w = w, x
        g.write(''.join(text))

def read_data(filename, words=False, max_length=None):
    with open(filename, 'r') as f:
        if words:
            data = ['␣'+w for w in f.read().split()]
        else:
            data = [x for x in '␣'.join(f.read().split())]
    if max_length is not None:
        data = data[:max_length]
    data, alphabet = bvmm.create_index(data)
    print('Sequence characters:', len(data))
    print('Distinct characters:', len(alphabet))
    return data, alphabet

def save_tree(root, filename):
    with open(filename, 'wb') as f:
        pickle.dump(root, f)

def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

#%%
# preprocess('dat/text_dickens.txt.bak', 'dat/text_dickens.txt')
# preprocess('dat/text_keats.txt.bak', 'dat/text_keats.txt')

#%% Keats by character =========================================================
%%time
data, alphabet = read_data('dat/text_keats.txt', words=False)
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
# save_tree(mcmc, 'out/text_1c.pickle')
# # mcmc = load_tree('out/text_1c.pickle')
# bvmm.write_tree(mcmc, alphabet, 'out/text_1c.net', min_samples=0.1)

#%% Keats by word ==============================================================
%%time
data, alphabet = read_data('dat/text_keats.txt', words=True)
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
# save_tree(mcmc, 'out/text_1w.pickle')
# # mcmc = load_tree('out/text_1w.pickle')
# bvmm.write_tree(mcmc, alphabet, 'out/text_1w.net', min_samples=0.1)

#%% Dickens by character =======================================================
# %%time
data, alphabet = read_data('dat/text_dickens.txt', words=False)
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
# save_tree(mcmc, 'out/text_2c.pickle')
# # mcmc = load_tree('out/text_2c.pickle')
# bvmm.write_tree(mcmc, alphabet, 'out/text_2c.net', min_samples=0.1)

#%%
# def sample_counts(v, cs):
#     cs.append(v.sample_count)
#     return cs
# cs = bvmm.io._visit_valid(sample_counts, mcmc, min_samples=1e-16, args=[])

#%% Dickens by word ============================================================
%%time
filename = 'dat/text_dickens.txt'
data, alphabet = read_data(filename, words=True)
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
save_tree(mcmc, 'out/text_2w.pickle')
# mcmc = load_tree('out/text_2w.pickle')
bvmm.write_tree(mcmc, alphabet, 'out/text_2w.net', min_samples=0.1)

# %%
