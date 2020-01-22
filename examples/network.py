#%%
import bvmm
import pickle

#%%
def read_data(filename, max_length=None):
    with open(filename, 'r') as f:
        data = []
        for l in f:
            v, w = l.split()[:2]
            data.append(('-'+v, '-'+w))
    if max_length is not None:
        data = data[:max_length]
    data, alphabet = bvmm.create_index(data, kind='network')
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
%%time
data, alphabet = read_data('dat/network_radoslaw.txt')
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10, kind='network')
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
# save_tree(mcmc, 'out/network_radoslaw.pickle')
# # mcmc = load_tree('out/network_radoslaw.pickle')
# bvmm.write_tree(mcmc, alphabet, 'out/network_radoslaw.net', min_samples=0.1)

# %%
