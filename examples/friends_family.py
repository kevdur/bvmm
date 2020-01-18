#%%
import bvmm

#%%
# Read the network data in as a list of pairs.
with open('dat/friends_family.csv', 'r') as f:
    data, kind = [tuple(l.split()[:2]) for l in f], 'network'
data, alphabet = bvmm.create_index(data, kind=kind)

# %%
%%time
# Run an MCMC simulation (55 min.).
root, counts = bvmm.mcmc(data, alphabet, 1_000_000, period=10,
                         prior='poisson', kind=kind)

# %%
print(counts)

# %%
bvmm.print_tree(root, alphabet, min_samples=.1, max_counts=3)

#%%
bvmm.write_tree(root, alphabet, 'dat/friends_family.net', min_samples=.1)

# %%
# import pickle
# with open('dat/friends_family.pickle', 'wb') as f:
#     pickle.dump((root, counts), f)
