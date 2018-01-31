
def waveform_difference_cc(obs, syn):
    nt = obs.stats.number_samples
    dt = obs.stats.sample_rate

    obs = obs.data
    syn = syn.data

    ioff = (np.argmax(cc)-nt+1)*dt
    if ioff <= 0:
        wrsd = syn[ioff:] - obs[:-ioff]
    else:
        wrsd = syn[:-ioff] - obs[ioff:]

    return np.sqrt(np.sum(wrsd*wrsd*dt))


