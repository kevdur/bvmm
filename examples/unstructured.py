#%%
import bvmm
import numpy as np
import pandas as pd
from math import log
from scipy.special import gammaln

#%%
k, full = 10, False
alphabet = [chr(i) for i in range(97, 97+k)]
root = bvmm.rand_tree(1, alphabet)
root.counts[:] = [1/k] * k
bvmm.print_tree(root, alphabet, max_counts=5)

# %%
ps = {}
for n in (10, 100, 1000, 2000, 5000, 10_000):
    nps = []
    for i in range(20):
        data = bvmm.rand_data(root, n)
        mcmc, counts = bvmm.mcmc(data, alphabet, 10_000, 10, full=full)
        nps.append((mcmc, counts))
    ps[n] = nps

#%%
df = pd.DataFrame(columns=['x', 'p', 'n', 'i'])
for n, trees in ps.items():
    for i, (r, c) in enumerate(trees):
        data = zip(alphabet, [w.sample_count for w in r.children])
        ndf = pd.DataFrame(data, columns=['x', 'p'])
        ndf['n'], ndf['i'] = n, i
        df = pd.concat((df, ndf), sort=False)
df.to_csv('dat/unstructured_{}.csv'.format(k), index=False)

#%% Asymptotic =================================================================
n = 1_000_000
alphabet = ['0', '1']
data = np.random.randint(0, 2, n)
root = bvmm.tree.create_tree(1, data, alphabet)
alpha = np.ones(2)
args = (alpha, bvmm.likelihood.luniform_ratio, bvmm.tree.Options(full=True))
bvmm.print_tree(root, alphabet, full=True, verbose=True)
print('Likelihood:', bvmm.likelihood._llhd(root, data, alphabet, *args))

# %%
bvmm.tree.activate(root, data, alphabet, opts)
bvmm.print_tree(root, alphabet, full=True, verbose=True)
print('Likelihood:', bvmm.likelihood._llhd(root, data, alphabet, *args))

# %%
def f(n, alpha=1):
    return 0.5*log(2*np.pi) -2*gammaln(alpha) + gammaln(2*alpha) \
        + (1-alpha)*log(4) - 0.5*log(n)

# %%
ns = np.arange(100, 100_000, 100)
xs = [f(n) for n in ns]
ls = []
data = np.random.randint(0, 2, ns[-1])
for n in ns:
    root = bvmm.tree.create_tree(1, data[:n], alphabet)
    bvmm.tree.activate(root, data[:n], alphabet, opts)
    ls.append(bvmm.likelihood._llhd(root, data[:n], alphabet, *args))
df = pd.DataFrame(dict(n=ns, x=xs, l=ls))
df.to_csv('dat/unstructured_asymptotic.csv', index=False)

# %%
