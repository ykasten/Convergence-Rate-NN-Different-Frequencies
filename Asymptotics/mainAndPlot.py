from decimal import *
from math import factorial as fact
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
from scipy.io import savemat
from scipy.linalg import eigh
from scipy.linalg import norm
from scipy.special import comb as choose_exact
import time
import sys

# plotting parameters
fs = 46
matplotlib.rcParams.update({'font.size' : fs})
matplotlib.rcParams.update({'axes.grid' : True})
matplotlib.rcParams.update({'axes.labelweight' : 'bold'})
fgs = [12,8]

max_float = sys.float_info.max
min_float = sys.float_info.min
min_num = 10**-1000
max_num = 10**1000
getcontext().prec = 1000
getcontext().Emax = MAX_EMAX
getcontext().Emin = -MAX_EMAX

def choose(n, k):
    # compute n choose k
    return Decimal(choose_exact(n, k, exact=True))

def beta(p, k, q, d = -1):
    # compute c2(q,d,k)/k!
    p = int(p)
    k = int(k)
    q = int(q)
    beta = (-1)**q * choose(p,q) * choose(2*q, k)
    return beta

def gamma(d,k):
    # compute c1(d,k)*k!
    gamma = Decimal(np.pi)**(Decimal(d/2)) * 2 / d * (-1)**k / (choose(int(k + (d-2)/2) , k) * fact(int((d-2)/2))) / 2**k
    return gamma

def cdk(d, k):
    # compute cdk
    d = int(d)
    k = int(k)
    p = int(k + (d-2) / 2)
    cdk_pos = 0
    cdk_neg = 0
    min_n1 = 2*(int(np.ceil(k/2))) - k + 1
    for q in range(int(np.ceil(k/2)), p+1):
        n1 = Decimal(2*q - k + 1)
        n2 = Decimal(2*q - k + 2)
        if k % 2 == 0:
            rq = (-2**n2 / (2*n1) + (2**n2 - choose(n2, int(n2/2))) / (2*n2)) / 2**(n1 + 1 - min_n1)
        else:
            rq = (2**n1 - choose(n1, int(n1/2))) / (2*n1) / 2**(n1 - min_n1)

        bet = beta(p, k, q, d)
        if bet > 0:
            cdk_pos += rq * bet
        else:
            cdk_neg += rq * bet

    cdk = (cdk_pos + cdk_neg) * gamma(d, k) / 2**(min_n1) / 2
    return cdk

def rate(d, k):
    return float(1/cdk(d,k))

if len(sys.argv) >= 4:
    max_d = int(sys.argv[1])
    min_k = int(sys.argv[2])
    max_k = int(sys.argv[3])
    print('ds = [2,4,6,...,%d], ks = [%d,...,%d]' % (max_d, min_k, max_k))
else:
    max_d = 100
    min_k = 800
    max_k = 1000
    print('ds = [2,4,6,...,%d], ks = [%d,...,%d]' % (max_d, min_k, max_k))
if len(sys.argv) >= 5:
    save_plot = sys.argv[4]
else:
    save_plot = True

ks = np.arange(min_k, max_k+1, 1)
ds = np.arange(2, max_d+1, 2)

cdks = np.zeros((len(ds), len(ks)))
rates = np.zeros((len(ds), len(ks)))

for i_d, d in enumerate(ds):
    print('d = %d' % d)
    for i_k, k in enumerate(ks):
        #print('*** d = %d, k = %d' % (d, k))
        res = cdk(d, k)
        #print(res)
        cdks[i_d, i_k] = float(res)
        if res == 0:
            rates[i_d, i_k] = np.nan
        else:
            rates[i_d, i_k] = float(1/res)
        if np.isnan(float(rates[i_d, i_k])) or np.isinf(float(rates[i_d, i_k])):
            print('*** ERROR: d = %d, k = %d, cdk = %.4f' % (d, k, res))
print('done!')

plt.figure(figsize = fgs)

gds = np.zeros(len(ds))
for i_d, d in enumerate(ds):
    d = int(d)
    coeffs1 = np.polyfit(np.log(ks), np.log(rates[i_d, :].reshape(-1)), 1)
    gds[i_d] = coeffs1[0]

plt.subplot(1, 1, 1)
plt.plot(ds, gds, 'o-b', linewidth = 2, markersize = 10, color = 'mediumblue')
plt.xlabel(r'$d$')
plt.ylabel(r'$g(d)$')
xt = [0, max_d/2, max_d]
yt = xt
plt.yticks(yt)
plt.gcf().subplots_adjust(bottom=0.3, left = .2)

if save_plot:
    plt.savefig('figures/cdk_slope_%d_%d_%d.jpg' % (max_d, min_k, max_k), format='jpg')
    savemat('../mat_plotting/results/cdk_plot_%d_%d_%d.mat' % (max_d, min_k, max_k), {'ds' : ds, 'gds' : gds})
    print('saved!')
