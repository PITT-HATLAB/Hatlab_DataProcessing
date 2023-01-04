from typing import Tuple, Any, Optional, Union, Dict, List

import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import json
from scipy.optimize import minimize_scalar

from Hatlab_DataProcessing.base import Analysis, AnalysisResult
from Hatlab_DataProcessing.fitter.generic_functions import Lorentzian
from Hatlab_DataProcessing.analyzer.cut_peak import cut_peak

plt.close("all")

data = json.load(open("D:\\Temp\\Q_0.825mA_freq_3520_3520-2processed"))

x_data = np.array(data["biasList"])
freq_list = np.array(data["freqList"])
q_data = np.array(data["glist"])[:,:,0]



# func params
n_modes = 2
fit_tol = 0.15
n_peak_bw = 30

# func body
results = np.zeros((n_modes, len(x_data))) + np.nan

for i, data in enumerate(tqdm(q_data)):
    new_data = data
    for n_ in range(n_modes):
        fit_ = Lorentzian(freq_list, new_data)
        fit_result_ = fit_.run(nan_policy="omit")
        A_ = fit_result_.params["A"].value
        k_ = fit_result_.params["k"].value
        x0_ = fit_result_.params["x0"].value
        of_ = fit_result_.params["of"].value
        # fit_result_.plot()

        # check if fitting is successful, if True, add fitted data to result list
        fit_success = fit_result_.success and (fit_result_.params["A"].stderr/np.abs(A_) < fit_tol)
        if fit_success:
            results[n_, i] = x0_
            # remove current fitted peak and get ready for the next fitting
            peak_start_idx = np.argmin(np.abs(freq_list-x0_+np.sqrt(n_peak_bw/k_)))
            peak_end_idx = np.argmin(np.abs(freq_list-x0_-np.sqrt(n_peak_bw/k_)))
            peak_region = np.clip([peak_start_idx, peak_end_idx], 0, len(freq_list))
            temp_ = np.arange(0, len(new_data))
            new_data = np.where((temp_ > peak_region[0]) & (temp_ < peak_region[1]), np.nan,
                                new_data)
        else:
            break


#todo: split the frequnecy traces
results = np.zeros((n_modes, len(x_data))) + np.nan
trace0 = np.linspace(0, -90, 91)
trace1 = np.linspace(0, -90, 91) + 50

results[0, 0: 50] = trace0[0:50]
results[0, 60:] = trace1[60:]
results[1, 30:61] = trace1[30:61]

data = results

plt.close("all")

correct_data = np.zeros_like(data)
last_data = data[:,0]
correct_data[:, 0] = last_data
for i in range(1, len(x_data)):
    new_data = data[:, i]
    for j, d in enumerate(last_data):
        # print(i, j, d, new_data)
        diff = np.abs(d-new_data)
        try:
            correct_data[j, i] = new_data[np.nanargmin(diff)]
        except ValueError:
            correct_data[j, i] = np.nan
        print(i, j, d, new_data, diff, correct_data[j, i])

    last_data = new_data

        

plt.figure()
# plt.pcolormesh(x_data, freq_list, q_data.T)
for i in range(n_modes):
    plt.plot(results[i])
    plt.plot(correct_data[i], "*")


plt.figure()
# plt.pcolormesh(x_data, freq_list, q_data.T)
for i in range(n_modes):
    plt.plot(x_data, correct_data[i], "*")