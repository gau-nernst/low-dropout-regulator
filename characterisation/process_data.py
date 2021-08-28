import numpy as np

def process_id_vgs(i_d: np.ndarray, v_gs: np.ndarray):
    id_sqrt = np.sqrt(i_d)
    id_sqrt_grad = np.gradient(id_sqrt, v_gs)
    max_idx = np.argmax(id_sqrt_grad)
    
    slope = id_sqrt_grad[max_idx]
    v_t = v_gs[max_idx] - id_sqrt[max_idx] / slope

    return v_t, slope
