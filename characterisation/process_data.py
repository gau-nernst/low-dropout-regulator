import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

k_boltzman = 1.380649e-23
q_electron = 1.60217662e-19

def thermal_voltage(temp: float=300):
    return k_boltzman * temp / q_electron

def process_id_vgs(i_d: np.ndarray, v_gs: np.ndarray):
    id_sqrt = np.sqrt(i_d)
    id_sqrt_grad = np.gradient(id_sqrt, v_gs)
    max_idx = np.argmax(id_sqrt_grad)
    
    slope = id_sqrt_grad[max_idx]
    v_t = v_gs[max_idx] - id_sqrt[max_idx] / slope
    k = 2*slope**2

    return v_t, k

def process_body_effect(v_sb: np.ndarray, v_t: np.ndarray, v_to: float):
    def fitting_eqn(v_sb, gamma, two_phi):
        return gamma * (np.sqrt(two_phi + v_sb) - np.sqrt(two_phi))

    popt, pcov = curve_fit(fitting_eqn, v_sb, v_t - v_to)
    return popt

def process_id_vds(i_d: np.ndarray, v_ds: np.ndarray, v_gs: float, v_th: float):
    v_ov = v_gs - v_th
    for i, x in enumerate(v_ds):
        if x > v_ov:
            idx = i
            break

    slope = (i_d[-1] - i_d[idx]) / (v_ds[-1] - v_ds[idx])
    return slope / i_d[idx:].mean()

def process_ic_vbe(i_c: np.ndarray, v_be: np.ndarray, T: float=300):
    ln_ic = np.log(i_c)
    grad = np.gradient(ln_ic, v_be)
    idx = np.argmax(grad)

    slope = grad[idx]
    ln_Is = ln_ic[idx] - v_be[idx]*slope
    Is = np.exp(ln_Is)
    n = k_boltzman * T / q_electron / slope

    return Is, n

def process_subthreshold(i_d: np.ndarray, v_gs: np.ndarray, temp: float=300):
    ln_id = np.log(i_d)
    
    linreg = LinearRegression()
    for size in range(2, len(v_gs)):
        x = v_gs[:size].reshape(-1,1)
        y = ln_id[:size]
        linreg.fit(x, y)
        score = np.abs((linreg.predict(x)-y) / y[0]).mean()
        if score > 1e-3:
            break
    
    slope = linreg.coef_[0]
    n = 1/slope/thermal_voltage(temp)
    ln_is = linreg.intercept_
    i_s = np.exp(ln_is)
    
    return n, i_s
