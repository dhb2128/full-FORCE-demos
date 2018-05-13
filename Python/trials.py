import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def fullforce_oscillation_test(dt, showplots=False):
    dt_per_s = round(1 / dt)

    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0, 2, 2 * dt_per_s + 1), 1)
    omega = np.zeros((2 * dt_per_s + 1, 1))
    omega = np.linspace(2 * np.pi, 6 * np.pi, 1 * dt_per_s + 1)
    targ = np.zeros((2 * dt_per_s + 1, 1))
    targ[0:(1 * dt_per_s + 1), 0] = np.sin(t[0:(1 * dt_per_s + 1), 0] * omega)
    targ[1 * dt_per_s:(2 * dt_per_s + 1)] = - \
        np.flipud(targ[0:(1 * dt_per_s + 1)])

    # A simpler example: just a sine wave
    '''
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*omega)
    '''

    # A slightly harder example: sum of sine waves
    '''
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.ones((2*dt_per_s+1,1)) * 4 *np.pi
    targ = np.sin(t*2*omega) * np.sin(t*omega/4)
    '''
    inp = np.zeros(targ.shape)
    inp[0:round(0.05 * dt_per_s), 0] = np.ones((round(0.05 * dt_per_s)))
    hints = np.zeros(targ.shape)

    if showplots:
        plot_trial(inp, targ, hints)
    return inp, targ, hints


def fullforce_poisson_clicks(dt, L=None, max_rate=40, showplots=False):
    rv = beta(4, 4)
    dt_per_s = round(1 / dt)
    t = np.linspace(0, 2.2, 2.2 * dt_per_s + 1)

    # always have one side have a higher rate
    if L is None:
        samples = np.arange(max_rate)
        L = np.random.choice(samples[samples != max_rate / 2])
    R = max_rate - L

    click_dt = 0.02
    min_onset = 0.1
    min_onset_idx = round(min_onset / dt)
    min_offset = 0.75
    max_offset = 1.5
    resp_off_idx = round(2/dt)
    t_stim_on = np.random.randint(
        low=round(min_offset / click_dt), high=round(max_offset / click_dt))  # sec
    l_click = np.repeat(np.random.rand(t_stim_on) <
                        click_dt * L, round(click_dt / dt))
    r_click = np.repeat(np.random.rand(t_stim_on) <
                        click_dt * R, round(click_dt / dt))

    inp = np.zeros((len(t), 2))
    t_stim_off_idx = min_onset_idx + t_stim_on * round(click_dt / dt)
    inp[min_onset_idx:t_stim_off_idx, 0] = l_click
    inp[min_onset_idx:t_stim_off_idx, 1] = r_click
    hints = np.zeros((len(t), 1))
    hints[min_onset_idx + 1:t_stim_off_idx,
          0] = (np.cumsum(np.diff(l_click)) - np.cumsum(np.diff(r_click))) / (max_rate / 4)
    targ = np.ones_like(hints) * -0.5
    t_rv = np.linspace(0, 1,  resp_off_idx - t_stim_off_idx)
    targ[t_stim_off_idx:resp_off_idx, 0] = -0.5 + \
        np.sign(L - R) * 1.5 * rv.pdf(t_rv) / rv.pdf(t_rv).max()

    if showplots:
        plot_trial(inp, targ, hints)
    return inp, targ, hints, L, t_stim_off_idx


def plot_trial(inp, targ, hints):
    plt.figure()
    plt.plot(targ)
    plt.plot(hints)
    plt.plot(inp)
    plt.legend(['Target', 'Hints', 'Input'])
