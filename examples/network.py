#%%
import bvmm
import pickle

#%%
def read_data(filename, sep='-', max_length=None):
    with open(filename, 'r') as f:
        data = []
        for l in f:
            v, w = l.split()[:2]
            data.append((sep+v, sep+w))
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

#%% Friends
%%time
data, alphabet = read_data('dat/network_friends.txt', sep='_')
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10, kind='network')
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=3)
print(counts)
save_tree(mcmc, 'out/network_friends.pickle')
# mcmc = load_tree('out/network_friends.pickle')
filename = 'out/network_friends.net'
bvmm.write_tree(mcmc, alphabet, filename, min_samples=0.1)

#%% EU
%%time
data, alphabet = read_data('dat/network_eu.txt', sep='_')
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10, kind='network')
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=3)
print(counts)
save_tree(mcmc, 'out/network_eu.pickle')
# mcmc = load_tree('out/network_eu.pickle')
filename = 'out/network_eu.net'
bvmm.write_tree(mcmc, alphabet, filename, min_samples=0.1)

#%% Radoslaw
%%time
data, alphabet = read_data('dat/network_radoslaw.txt')
mcmc, counts = bvmm.mcmc(data, alphabet, 100_000, 10, kind='network')
bvmm.print_tree(mcmc, alphabet, min_samples=0.1, max_counts=5)
print(counts)
save_tree(mcmc, 'out/network_radoslaw.pickle')
# mcmc = load_tree('out/network_radoslaw.pickle')
filename = 'out/network_radoslaw.net'
bvmm.write_tree(mcmc, alphabet, filename, rooted=False, min_samples=0.1)

# %% Friends preprocessing
# calls = pd.read_csv('dat/network_friends.csv.bak', parse_dates=['local_time'])
# calls.rename(columns=dict(zip(calls.columns, ['A', 'B', 'time', 'type', 'duration', 'hash'])), inplace=True)
# calls.loc[calls.type == 'incoming+', 'type'] = 'incoming'
# calls.loc[calls.type == 'outgoing+', 'type'] = 'outgoing'
# calls.loc[calls.type == 'incoming', ['A', 'B']] = calls.loc[calls.type == 'incoming', ['B', 'A']].values
# calls = calls[~calls.A.isna() & ~calls.B.isna()].reset_index(drop=True)
# calls = calls[(calls.time >= datetime(2010, 11, 1)) & (calls.time < datetime(2011, 5, 1))]
# calls = calls.sort_values(['time', 'A', 'B'])
# calls = calls.drop_duplicates(subset=['time', 'A', 'B']).reset_index(drop=True)
# calls['timestamp'] = calls.time.apply(lambda t: t.timestamp())
# calls[['A', 'B', 'timestamp']].to_csv('dat/network_friends.txt', sep=' ', header=False, index=False)

# %%
