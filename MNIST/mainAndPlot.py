import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from optimize import optimize
import time

from cfg import cfg

save_fig = True
test_name = 'mnist_%d' % cfg['pi']
cfg['test_name'] = test_name

# optimization
print('*** Running test: %s\n' % test_name)
start = time.time()
optimize(cfg)
end = time.time()
print('\n*** Done training! Training time: %.3f minutes' % ((end - start)/60))

fs = 20
matplotlib.rcParams.update({'font.size' : fs})
matplotlib.rcParams.update({'axes.grid' : True})
matplotlib.rcParams.update({'axes.labelweight' : 'bold'})
lw = 3
fgs = [12,8]

all = np.load('results/' + test_name + '.npz')
train = all['train_acc']
test = all['test_acc']
val = all['val_acc']
cfg_ = all['cfg'].reshape(1)[0]

pi = cfg_['pi']
epochs = np.arange(len(train))
stop_ind = np.argmax(test)
stop_epoch = epochs[stop_ind]
M_test = test[stop_ind]
M_val = val[stop_ind]
expected_M_val = (100 - pi + pi/10) / 100
saturation_inds = np.where(train >= .8)
if len(saturation_inds[0]) > 10 :
    max_ind = saturation_inds[0][10]
else :
    max_ind = len(train)
print(max_ind)
pi_s = '{:d}'.format(pi)
M_test_s = '{:.3f}'.format(M_test)
M_val_s = '{:.3f}'.format(M_val)
emv_s = '{:.3f}'.format(expected_M_val)

plt.figure(figsize = fgs)
plt.gcf().subplots_adjust(bottom=.2, left = .2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(epochs[:max_ind], train[:max_ind], '-', color = 'r',  label='Train (corrupted)', linewidth = lw)
plt.plot(epochs[:max_ind], val[:max_ind], '-', color = 'g', label='Val (corrupted)', linewidth = lw)
plt.plot(epochs[:max_ind], test[:max_ind], '-', color = 'b', label='Test (uncorrupted)', linewidth = lw)
plt.plot((stop_epoch, stop_epoch), (0,1), ':', color = 'k', label='Early stopping point', linewidth = lw)
plt.legend(loc=4)
xt = np.arange(0, max_ind, 400)
plt.xticks(xt, xt)
yt = [0, .5, 1]
plt.yticks(yt, yt)

if save_fig:
    plt.savefig('figures/' + test_name + '.jpg', format='jpg')
    savemat('../mat_plotting/results/' + test_name + '.mat', {'epochs' : epochs[:max_ind],
                                        'train' : train[:max_ind],
                                        'val' : val[:max_ind],
                                        'test' : test[:max_ind],
                                        'stop_epoch' : stop_epoch,
                                       })
    print('saved!')
