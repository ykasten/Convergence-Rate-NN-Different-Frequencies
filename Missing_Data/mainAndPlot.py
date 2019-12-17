import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import os
import pandas as pd
from scipy.linalg import eigh, norm
from scipy.io import savemat
import sys
import time
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gen_data as gen_data_package

cfg = {
    # meta
    'save_losses' : False,

    # dataset
    'dim' : 2,
    'gen_x_func' : 'gen_x_circle_holes_',
    'gen_y_func' : 'gen_y_fourier',
    'add_phase' : False,
    'n_val' : 32,
    'n_train' : 32,
    'ks' : [6],
    'resample' : False,

    # network
    'n_hidden' : 1,
    'n_units' : 4000,
    'kappa' : 1,
    'hidden_bias' : 'zeros', # none/zeros/normal
    'outer_fixed' : False,
    'even_only' : False,
    'odd_only' : False,

    # optimization
    'eta' : 0.0001,
    'n_epochs_max' : 300000,
    'n_batch' : 0,
    'stop_threshold_percent' : .0001,
    'max_training_time_in_minutes' : 1200,
}

def sim(cfg):

    if cfg['n_batch'] == 0: cfg['n_batch'] = cfg['n_train']

    device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
    use_parallel_gpus = False
    print('\r\ndevice is: %s\r\n' % device)

    ##################################################################################

    # define network

    class Net(nn.Module):

        def __init__(self, d, n_sizes, hidden_weights_init_values, hidden_bias, kappa, outer_fixed, out_size = 1):
            super(Net, self).__init__()

            n_hidden = len(n_sizes)
            hidden_in_sizes = n_sizes[:]
            hidden_in_sizes.insert(0,d)
            hidden_out_sizes = n_sizes
            outer_in_size = n_sizes[-1]
            outer_out_size = out_size

            self.hidden = nn.ModuleList()
            self.hidden.extend([nn.Linear(hidden_in_sizes[i], hidden_out_sizes[i], bias=(hidden_bias!='none')) for i in range(n_hidden)])
            for i in range(n_hidden):
                # init normal weights in hidden layers
                #tr.nn.init.normal_(self.hidden[i].weight, mean=0, std=kappa)
                tr.nn.init.kaiming_normal_(self.hidden[i].weight, a=np.sqrt(5))

                # init either 0's or normal biases in hidden layers
                if hidden_bias == 'zeros':
                    self.hidden[i].bias.data = tr.zeros([hidden_out_sizes[i]])
                elif hidden_bias == 'normal':
                    std = 1 / np.sqrt(3*hidden_in_sizes[i])
                    tr.nn.init.normal_(self.hidden[i].bias, mean=0, std=std)

            if outer_fixed:
                # set requires_grad to False, and initialize with values of +-1
                self.outer = nn.Linear(outer_in_size, outer_out_size, bias=False)
                for param in self.outer.parameters():
                    param.requires_grad = False
                self.outer.weight.data = tr.from_numpy(np.sign(rn.uniform(-1, 1, [outer_out_size, outer_in_size])) / np.sqrt(outer_in_size)).float()
            else:
                # initialize with normal
                self.outer = nn.Linear(outer_in_size, outer_out_size, bias=False)
                tr.nn.init.kaiming_normal_(self.outer.weight, a=np.sqrt(5))

        def forward(self, x):
            for i in range(len(self.hidden)):
                x = F.relu(self.hidden[i](x))
            x = self.outer(x)
            return x

    class XYDataset():
        def __init__(self, data):
            self.x = tr.from_numpy(data['x']).float().view(len(data['x']), -1)
            self.y = tr.from_numpy(data['y']).float().view(len(data['y']), -1)
            self.vals = data['vals']
            self.theta = tr.from_numpy(data['theta']).float().view(len(data['x']), -1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.theta[idx]

    def gen_data(cfg, pr=False):
        data_train = gen_data_package.gen_xy(cfg, 'train')
        inds = np.argsort(data_train['theta'])
        if cfg['gen_y_func'] == 'gen_y_H_inf' or cfg['gen_y_func'] == 'gen_y_H_0' or cfg['gen_y_func'] == 'gen_y_H_inf_norm_1':
            inds = rn.permutation(cfg['n_train'])
            inds = inds[:cfg['n_train']]
            data_val = {'x' : data_train['x'][inds, ...], 'y' : data_train['y'][inds], 'vals' : data_train['vals'], 'theta' : data_train['theta'][inds]}
        else:
            data_val = gen_data_package.gen_xy(cfg, 'val')
        inds = np.argsort(data_train['theta'])
        data_train['theta'] = data_train['theta'][inds]
        data_train['x'] = data_train['x'][inds,:]
        data_train['y'] = data_train['y'][inds]
        inds = np.argsort(data_val['theta'])
        data_val['theta'] = data_val['theta'][inds]
        data_val['x'] = data_val['x'][inds,:]
        data_val['y'] = data_val['y'][inds]

        trainset = XYDataset(data_train)
        valset = XYDataset(data_val)

        return {'trainset' : trainset, 'valset' : valset, 'W' : data_train['W']}

    def gen_net(cfg, W, device, use_parallel_gpus = False, pr = False):
        net = Net(d=cfg['dim'],
                    n_sizes=[cfg['n_units'] for i in range(cfg['n_hidden'])],
                    hidden_weights_init_values=W,
                    hidden_bias=cfg['hidden_bias'],
                    kappa=cfg['kappa'],
                    outer_fixed=cfg['outer_fixed'])
        if pr: print('\r\nnet:\r\n', net)
        if tr.cuda.device_count() > 1 and use_parallel_gpus:
            print("Let's use", tr.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        net.to(device)
        return net

    def optimize(cfg, device, net, datasets, pr = False):

        # gen data
        trainset = datasets['trainset']
        valset = datasets['valset']

        theta_train = trainset.theta.numpy().reshape(-1)
        y_train = trainset.y.numpy().reshape(-1)
        cfg_data_plot = cfg.copy()
        cfg_data_plot['gen_x_func'] = 'gen_x_circle_regular'
        cfg_data_plot['n_train'] = 10000
        data_plot = gen_data_package.gen_xy(cfg_data_plot, 'train')
        theta_plot = data_plot['theta']
        plot_sort_inds = np.argsort(theta_plot)
        x_plot = data_plot['x'][plot_sort_inds, :]
        x_plot_input = tr.from_numpy(x_plot).float().view(len(x_plot), -1)
        theta_plot = data_plot['theta'][plot_sort_inds]
        y_plot = data_plot['y'][plot_sort_inds]

        # create optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr = cfg['eta'])

        epoch_loss = np.zeros(cfg['n_epochs_max'])
        train_loss = np.zeros(cfg['n_epochs_max'])
        val_loss = np.zeros(cfg['n_epochs_max'])

        with tr.no_grad():
            val_output_before_training = net(valset.x.to(device))
            val_loss_before_training = criterion(val_output_before_training, valset.y.to(device)) * cfg['n_val'] / 2
            train_output_before_training = net(trainset.x.to(device))
            train_loss_before_training = criterion(train_output_before_training, trainset.y.to(device)) * cfg['n_train'] / 2
        print('\r\ninitial val/train loss is %.2f/%.2f\r\n' % (val_loss_before_training, train_loss_before_training))

        # train
        converged = False
        stopping_criterion = ''
        n_batches = cfg['n_train'] // cfg['n_batch']
        total_training_time = time.time()
        total_optimization_time = 0
        total_iter_time = 0
        total_sampling_time = 0
        total_compute_losses_time = 0
        other_time = 0
        epoch = 0

        while True:

            total_sampling_time -= time.time()
            if cfg['resample']:
                epoch_datasets = gen_data(cfg, pr=False)
                trainset = epoch_datasets['trainset']
                #trainloader = epoch_datasets['trainloader']
            total_sampling_time += time.time()

            total_iter_time -= time.time()
            epoch_Is = rn.permutation(cfg['n_train'])
            total_iter_time += time.time()
            accumulated_loss = .0

    #        for i, data in enumerate(trainloader):
            for i_batch in range(n_batches):
                # get the inputs
                total_iter_time -= time.time()
                batch_Is = epoch_Is[i_batch*cfg['n_batch']:(i_batch+1)*cfg['n_batch']]
                x_train_batch, y_train_batch = trainset.x[batch_Is], trainset.y[batch_Is]
                x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
                total_iter_time += time.time()

                total_optimization_time -= time.time()
                # zero the parameter gradients
                optimizer.zero_grad()   # zero the gradient buffers

                # forward + backward + optimize
                output = net(x_train_batch)
                loss = criterion(output, y_train_batch) * cfg['n_batch'] / 2
                loss.backward()
                optimizer.step()
                total_optimization_time += time.time()

                # print statistics
                accumulated_loss += loss.item()

            total_compute_losses_time -= time.time()
            epoch_loss[epoch] = accumulated_loss / n_batches
            with tr.no_grad():
                val_output = net(valset.x.to(device))
                val_loss[epoch] = criterion(val_output, valset.y.to(device)) * cfg['n_val'] / 2
                train_output = net(trainset.x.to(device))
                train_loss[epoch] = criterion(train_output, trainset.y.to(device)) * cfg['n_train'] / 2
            total_compute_losses_time += time.time()

            epoch += 1
            if pr and epoch % 100 == 0:
                print('epoch %7d val/train loss: %.2f/%.2f (%.2f%% for training)\r\n' %
                      (epoch, val_loss[epoch-1], train_loss[epoch-1], train_loss[epoch-1] / cfg['zero_train_loss'] * 100))
            if train_loss[epoch-1] / cfg['zero_train_loss'] < cfg['stop_threshold_percent'] / 100:
                print('\r\n@@@ after %d epochs, training loss is %.2f, reached stop threshold of %.2f%% and training is done\r\n'
                      % (epoch, train_loss[epoch-1], cfg['stop_threshold_percent']))
                converged = True
                stopping_criterion = 'converged'
                break
            training_time_so_far_in_minutes = (time.time() - total_training_time) / 60
            if training_time_so_far_in_minutes > cfg['max_training_time_in_minutes']:
                print('\r\nafter %d epochs, training time exceeded %d minutes threshold and training is stopped\r\n'
                      % (epoch, cfg['max_training_time_in_minutes']))
                stopping_criterion = 'time_out'
                break
            if np.isnan(epoch_loss[epoch-1]):
                print('\r\nafter %d epochs, epoch loss is NaN and training is stopped\r\n'
                     % epoch)
                stopping_criterion = 'reached_nan'
                break
            if epoch == cfg['n_epochs_max']:
                print('\r\nreached maximal number of epochs %d and training is stopped\r\n'
                     % cfg['n_epochs_max'])
                stopping_criterion = 'epochs_over'
                break

        #total_compute_losses_time += time.time()
        total_training_time = time.time() - total_training_time
        print('\r\nfinished training! training time: %.2f minutes (optimization time: %.2f%%, iter time: %.2f%%, compute train/val loss time: %.2f%%, sampling time: %.2f%%\r\n)'
              % (total_training_time/60,
              total_optimization_time/total_training_time*100,
              total_iter_time/total_training_time*100,
              total_compute_losses_time/total_training_time*100,
              total_sampling_time/total_training_time*100))

        with tr.no_grad():
            val_output_after_training = net(valset.x.to(device))
            val_loss_after_training = criterion(val_output_after_training, valset.y.to(device)) * cfg['n_val'] / 2
            train_output_after_training = net(trainset.x.to(device))
            train_loss_after_training = criterion(train_output_after_training, trainset.y.to(device)) * cfg['n_train'] / 2
            plot_output_after_training = net(x_plot_input.to(device))
        print('\r\nfinal val/train loss is %.2f/%.2f\r\n (%.2f%% for training)' % (val_loss_after_training, train_loss_after_training,
            train_loss_after_training * 100))

        return {'num_epochs' : epoch,
                'final_train_loss' : train_loss_after_training.numpy().flatten()[0],
                'final_val_loss' : val_loss_after_training.numpy().flatten()[0],
                'converged' : converged,
                'training_time' : total_training_time,
                'stopping_criterion' : stopping_criterion,
                'theta_train' : theta_train,
                'y_train' : y_train,
                'train_output' : train_output_after_training.numpy().reshape(-1),
                'theta_plot' : theta_plot,
                'plot_output' : plot_output_after_training.numpy().reshape(-1),
               }

    results = pd.DataFrame()

    if cfg['add_phase']:
        cfg['phases'] = rn.uniform(-np.pi, np.pi, len(cfg['ks']))
    else:
        cfg['phases'] = []
    print('\r\ntest cfg: \r\n', cfg, '\r\n')

    total_loops_time = time.time()

    pr = True

    # generate new dataset
    datasets = gen_data(cfg, pr=False)
    cfg['zero_train_loss'] = tr.sum(datasets['trainset'].y**2).cpu().numpy() / 2
    if pr: print('zero train loss is: %.3f\r\n' % cfg['zero_train_loss'])

    # create new network
    net = gen_net(cfg, datasets['W'], device, use_parallel_gpus, pr)

    # optimize and save results
    my_dict = {'ks' : cfg['ks'], 'phases' : cfg['phases'], 'eigenvalues' : datasets['trainset'].vals}
    my_dict.update(optimize(cfg, device, net, datasets, pr))
    results = results.append(my_dict, ignore_index = True)

    total_loops_time = (time.time() - total_loops_time) / 60
    print('\r\nfinished looping! total looping time: %.2f minutes\r\n' % total_loops_time)

    print(results)
    print('\r\nstopping criterion: %s' % results['stopping_criterion'])

    return results

# calling the sim
results = sim(cfg)
name = 'missing_data_k%d_n%d_epochs%d.pkl' % (cfg['ks'][0], cfg['n_train'], results['num_epochs'][0])

results.to_pickle('results/' + name)

# plotting
fs = 22
matplotlib.rcParams.update({'font.size': fs})

train_output = results['train_output'][0]
theta_train = results['theta_train'][0]
y_train = results['y_train'][0]
theta_plot = results['theta_plot'][0]
plot_output = results['plot_output'][0]

plt.figure(figsize = [12,6])
plt.plot(theta_plot, plot_output, '-', label = 'network output', color = 'darkorange', linewidth = 3.5)
plt.plot(theta_train, y_train, '.', label = 'training points', markersize = 24, color = '.25', markeredgecolor = 'k')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$f(\theta;W,\mathbf{a})$')
plt.gcf().subplots_adjust(bottom = .18)

plt.savefig('figures/' + name + '.jpg', format='jpg')
savemat('../mat_plotting/results/missing_data.mat', {'plot_theta' : theta_plot, 'plot_output' : plot_output,
                                        'train_theta' : theta_train, 'train_y' : y_train})
