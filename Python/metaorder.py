import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM, fbm, fgn
import pandas as pd

Nt = 100
T = 1
N_samples = 1000
hurst = 0.75
increments = np.empty((N_samples, Nt))
time_interval, dt = np.linspace(0, T, num=Nt, retstep=True)

for sample_index in range(N_samples):
    increments[sample_index, :] = fgn(n=Nt, hurst=hurst, length=T)
increments = pd.DataFrame(increments.T)
# increments.plot()
# plt.plot(time_interval, increments.T)
# plt.show()


def autocovariance_t(dataframe, tau):
    return (dataframe * dataframe.shift(-tau)).mean(axis=1)/(dt**(2*hurst))
covariogram = np.vectorize(lambda tau : autocovariance_t(increments, tau).mean())

Ntau = Nt//3
taus_array = np.arange(Ntau)
# autocovs = np.zeros((Nt, Ntau))
# for index, tau in enumerate(taus_array):
#     tau_covs = autocovariance_t(increments, tau)
#     autocovs[:tau_covs.size, index] = tau_covs
# plt.plot(time_interval, autocovs[:, :10])
# plt.show()
covariances = covariogram(taus_array)
plt.plot(taus_array, covariances)
plt.show()