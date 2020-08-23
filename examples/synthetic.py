#%%
import bvmm
import numpy as np
import pickle

#%%
def read_data(filename, words=False, sep='_', max_length=None):
    with open(filename, 'r') as f:
        if words:
            data = [sep+w for w in f.read().split()]
        else:
            data = [x for x in sep.join(f.read().split())]
    if max_length is not None:
        data = data[:max_length]
    data, alphabet = bvmm.create_index(data)
    print('Sequence characters:', len(data))
    print('Distinct characters:', len(alphabet))
    return data, alphabet

#%% Binary model ===============================================================
alphabet = ['0', '1']

#%%
# root = bvmm.rand_tree(5, alphabet)
# bvmm.print_tree(root, alphabet)

#%%
opts = bvmm.tree.Options()
root = bvmm.tree.create_tree(2, [], alphabet)
for v in [root, root.children[0], root.children[1],
          root.children[1].children[0], root.children[1].children[1]]:
    v.counts = np.zeros(2)
for v in [root, root.children[0], root.children[1],
          root.children[1].children[0], root.children[1].children[1]]:
    bvmm.tree.activate(v, [], alphabet, opts)
root.counts = np.array([0.02, 0.98])
root.children[0].counts = np.array([0.19, 0.81])
root.children[1].counts = np.array([0.27, 0.73])
root.children[1].children[0].counts = np.array([0.45, 0.55])
root.children[1].children[1].counts = np.array([0.64, 0.36])
bvmm.print_tree(root, alphabet, verbose=True)

# %%
n = 10_000
data = bvmm.rand_data(root, n)
# with open('dat/synthetic_1_{}.txt'.format(n), 'w') as f:
#     f.write(''.join(bvmm.apply_alphabet(data, alphabet)))

# %%
mcmc, counts = bvmm.mcmc(data, alphabet, 10_000, 10, full=True)
bvmm.print_tree(mcmc, alphabet, full=True, min_samples=.01)
print(counts)

#%% Ternary model ==============================================================
alphabet = ['a', 'b', 'c']
root = bvmm.rand_tree(10, alphabet)
bvmm.print_tree(root, alphabet)

#%%
n = 10_000
data = bvmm.rand_data(root, n)
# with open('dat/synthetic_2_{}.txt'.format(n), 'w') as f:
#     f.write(''.join(bvmm.apply_alphabet(data, alphabet)))

#%%
# data, alphabet = read_data('dat/synthetic_2_10000.txt', sep='')
mcmc, counts = bvmm.mcmc(data, alphabet, 10_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=.01)
print(counts)

# %% 25-character model ========================================================

alphabet = ['-{}'.format(x) for x in range(25)]
with open('dat/synthetic_3.pkl', 'rb') as f:
    root = pickle.load(f)
bvmm.print_tree(root, alphabet, max_counts=4)

# root = bvmm.rand_tree(50, alphabet)
# bvmm.print_tree(root, alphabet, max_counts=4)
# # with open('dat/synthetic_3.pkl', 'wb') as f:
# #     pickle.dump(root, f)

# %%
n = 10_000
with open('dat/synthetic_3_{}.txt'.format(n), 'r') as f:
    data = ['-' + x for x in f.read().split('-') if x]
data, alphabet = bvmm.create_index(data)

# data = bvmm.rand_data(root, n)
# # with open('dat/synthetic_3_{}.txt'.format(n), 'w') as f:
# #     f.write(''.join(bvmm.apply_alphabet(data, alphabet)))

# %%
%%time
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10)
bvmm.print_tree(mcmc, alphabet, min_samples=.01, max_counts=4)
print(counts)
bvmm.write_tree(mcmc, alphabet, 'dat/synthetic_3_{}.net'.format(n),
                min_samples=.01)

# %%
def activate_same(v1, v2):
    print(''.join(str(x) for x in bvmm.path_to(v1)), ''.join(str(x) for x in bvmm.path_to(v2)))
    for w1, w2 in zip(v1.children, v2.children):
        if w1.is_active:
            bvmm.tree.activate(w2, data, alphabet, bvmm.tree.Options())
            activate_same(w1, w2)
    
while mcmc.node_count > 1:
    bvmm.tree.deactivate(bvmm.tree.leaf(mcmc, 0))
activate_same(root, mcmc)
bvmm.print_tree(mcmc, alphabet, verbose=True, min_samples=0.01, max_counts=4)

# %%

# %%
