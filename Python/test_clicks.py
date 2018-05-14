import os
import sys
import joblib
import numpy as np
from trials import fullforce_poisson_clicks
from tqdm import trange
import pandas as pd

fn_time = "1526315369"
fn = os.path.join('data', fn_time +
                  '_fullforce_poisson_rnn.p.z')

if os.path.exists(fn):
    rnn = joblib.load(fn)
else:
    print("File not found")
    sys.exit()

inp, targ, = fullforce_poisson_clicks(dt=0.001)[:2]
nt = inp.shape[0]
N = 100  # trials

inps = np.zeros((nt, inp.shape[1], N))
targs = np.zeros((nt, N))
L_all = np.zeros(N)
t_stims = np.zeros_like(L_all)

# get trials
for i in trange(N):
    inps[:, :, i], targ, _, L_all[i], t_stims[i] = fullforce_poisson_clicks(
        dt=0.001)
    targs[:, i] = targ.ravel()

rnn.train_stats = None
norms, decisions = rnn.test_batch(inps, targs, t_stims,
                                  norm_only=True, norm_idx=[200, 2000],
                                  inps_and_targs=fullforce_poisson_clicks)


df = pd.DataFrame(data=np.stack((L_all, t_stims, norms, decisions), axis=1),
                  columns=['L', 'dur', 'norm', 'decision'])
# df['dur_bin'] = pd.cut(df.dur, 6, labels=False)

df_fn = os.path.join('data', fn_time + '_poisson_data.pkl')
df.to_pickle(df_fn)
