import numpy.random as rn
import numpy as np



def gen_xy(cfg, type):
    if type == 'train':
        n = cfg['n_train']
    elif type == 'val':
        n = cfg['n_val']
    else:
        print('bad type')
        return

    if cfg['gen_x_func'] == 'gen_x_arora':
        x, theta = gen_x_arora(n)

    if cfg['gen_y_func'] == 'gen_y_arora':
        y = gen_y_arora(theta, cfg['ks'], cfg['phases'])

    relevant = abs(y) > (2 / 3)
    x=x[relevant]
    y=y[relevant]
    y[y>0]=1
    y[y<0]=0
    y=y.astype(int)
    return {'x' : x, 'y' : y}






def gen_x_arora(n):
    theta = rn.uniform(-np.pi, np.pi, n)
    x = np.stack([np.cos(theta), np.sin(theta)]).T
    return x, theta

def gen_y_arora(theta, ks, phases = []):
    if phases == []:
        phases = np.zeros(len(ks))
    n = len(theta)
    y = np.zeros(n)
    for i_k, k in enumerate(ks):
        phi_k = phases[i_k]
        y = y + np.sin(k*(theta + phi_k))
    return y
