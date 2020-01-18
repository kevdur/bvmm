#%%
import bvmm
import numpy as np

#%%
with open('dat/keats_melancholy.txt', 'r') as f:
    data, kind = ['␣' if x.isspace() else x for x in f.read()], 'sequence'
data, alphabet = bvmm.create_index(data, kind=kind)

# %%
mcmc, counts = bvmm.mcmc()