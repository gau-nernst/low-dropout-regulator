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
    
    print(f"Point of max slope: Vgs = {v_gs[max_idx]:.2f}")
    linreg = LinearRegression()
    point1 = max_idx
    point2 = max_idx + 1
    while True:
        score1 = 1e6
        score2 = 1e6
        if point1 > 0:
            x = v_gs[point1-1:point2].reshape(-1,1)
            y = id_sqrt[point1-1:point2]
            linreg.fit(x, y)
            score1 = np.abs(linreg.predict(x) - y).mean() / id_sqrt[max_idx]
        if score2 < len(v_gs):
            x = v_gs[point1:point2+1].reshape(-1,1)
            y = id_sqrt[point1:point2+1]
            linreg.fit(x, y)
            score2 = np.abs(linreg.predict(x) - y).mean() / id_sqrt[max_idx]
            
        if min(score1, score2) > 1e-3:
            break
    
        if score1 < score2:
            point1 -= 1
        else:
            point2 += 1

    x = v_gs[point1:point2].reshape(-1,1)
    y = id_sqrt[point1:point2]
    linreg.fit(x, y)
    print(f"Straight line: {v_gs[point1]:.2f}, {v_gs[point2]:.2f}")
    
    slope = linreg.coef_[0]
    v_t = -linreg.intercept_ / slope
    k = 2*slope**2

    return v_t, k

def process_body_effect(v_sb: np.ndarray, v_t: np.ndarray, v_to: float):
    def fitting_eqn(v_sb, gamma, two_phi):
        return gamma * (np.sqrt(two_phi + v_sb) - np.sqrt(two_phi))

    popt, pcov = curve_fit(fitting_eqn, v_sb, v_t - v_to)
    return popt

def process_id_vds(i_d: np.ndarray, v_ds: np.ndarray):
    linreg = LinearRegression()
    for size in range(2, len(v_ds)):
        x = v_ds[-size:].reshape(-1,1)
        y = i_d[-size:]
        linreg.fit(x, y)
        score = np.abs((linreg.predict(x)-y) / y[-1]).mean()
        if score > 1e-3:
            break

    print(f"Straight line: Vds = {v_ds[-size]}")

    i_d0 = linreg.intercept_
    lambda_ds = linreg.coef_[0] / i_d0
    
    return lambda_ds, i_d0

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
    
    print(f"Straight line: Vgs = {v_gs[size-1]}")
    
    slope = linreg.coef_[0]
    n = 1/slope/thermal_voltage(temp)
    ln_is = linreg.intercept_
    i_s = np.exp(ln_is)
    
    return n, i_s
