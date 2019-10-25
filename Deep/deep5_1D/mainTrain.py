import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import numpy as np
import numpy.random as rn
import pandas as pd


import gen_data as gen_data_package

### change here for other test
testname = 'deep5_euc_'
sysstr = sys.argv[1]
from config import cfg_uniform_arora as cfg


device = tr.device("cpu")
use_parallel_gpus = False



# define network

class Net(nn.Module):

    def __init__(self, d, n_sizes, out_size=1):
        super(Net, self).__init__()
        self.d = d
        self.n_sizes = n_sizes
        self.out_size = out_size

        n_sizes.insert(0, d)
        self.hidden = nn.ModuleList()
        self.hidden.extend([nn.Linear(n_sizes[i], n_sizes[i + 1], bias=True) for i in range(len(n_sizes) - 1)])
        self.out_layer = nn.Linear(n_sizes[-1], out_size, bias=False)


        tr.nn.init.kaiming_normal_(self.out_layer.weight, a=np.sqrt(5))

        for i in range(len(self.hidden)):
            print(i)
            tr.nn.init.kaiming_normal_(self.hidden[i].weight, a=np.sqrt(5))


    def forward(self, x):
        for i in range(len(self.hidden)):
            x = F.relu(self.hidden[i](x))
        x = self.out_layer(x)
        return x


class XYDataset():
    def __init__(self, data):
        self.x = tr.from_numpy(data['x']).float().view(len(data['x']), -1)
        self.y = tr.from_numpy(data['y']).float().view(len(data['y']), -1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def gen_data(cfg, pr=False):
    data_train = gen_data_package.gen_xy(cfg, 'train')
    data_val = gen_data_package.gen_xy(cfg, 'val')

    trainset = XYDataset(data_train)
    valset = XYDataset(data_val)

    return {'trainset': trainset, 'valset': valset}



def gen_net(cfg, device, use_parallel_gpus=False, pr=False):
    net = Net(d=cfg['dim'], n_sizes=[cfg['n_units'] for i in range(cfg['n_hidden'])])
    print('\nnet:\n', net)
    if tr.cuda.device_count() > 1 and use_parallel_gpus:
        print("Let's use", tr.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)
    return net


def optimize(cfg, device, net, datasets, pr=False):
    trainset = datasets['trainset']
    valset = datasets['valset']

    # create optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(net.parameters(), lr = cfg['eta'])
    optimizer = optim.SGD(net.parameters(), lr=cfg['eta'])
    epoch_loss = np.zeros(cfg['n_epochs'])
    train_loss = np.zeros(cfg['n_epochs'])
    val_loss = np.zeros(cfg['n_epochs'])

    with tr.no_grad():
        val_output_before_training = net(valset.x.to(device))
        val_loss_before_training = criterion(val_output_before_training, valset.y.to(device))
        train_output_before_training = net(trainset.x.to(device))
        train_loss_before_training = criterion(train_output_before_training, trainset.y.to(device))
    print('\rinitial val/train loss is %.2f/%.2f\r' % (val_loss_before_training, train_loss_before_training))

    # train
    converged = False
    n_batches = cfg['n_train'] // cfg['n_batch']
    total_training_time = time.time()
    total_optimization_time = 0
    total_iter_time = 0
    total_sampling_time = 0

    epoch = 0
    while True:
        total_sampling_time -= time.time()
        if cfg['resample']:
            epoch_datasets = gen_data(cfg, pr=False)
            trainset = epoch_datasets['trainset']
            # trainloader = epoch_datasets['trainloader']
        total_sampling_time += time.time()

        total_iter_time -= time.time()
        epoch_Is = rn.permutation(cfg['n_train'])
        total_iter_time += time.time()
        accumulated_loss = .0


        for i_batch in range(n_batches):
            # get the inputs
            total_iter_time -= time.time()
            batch_Is = epoch_Is[i_batch * cfg['n_batch']:(i_batch + 1) * cfg['n_batch']]
            x_train_batch, y_train_batch = trainset.x[batch_Is], trainset.y[batch_Is]
            x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
            total_iter_time += time.time()

            total_optimization_time -= time.time()
            # zero the parameter gradients
            optimizer.zero_grad()  # zero the gradient buffers

            # forward + backward + optimize
            output = net(x_train_batch)
            loss = criterion(output, y_train_batch)
            loss.backward()
            optimizer.step()
            total_optimization_time += time.time()

            # print statistics
            accumulated_loss += loss.item()

        epoch_loss[epoch] = accumulated_loss / n_batches

        val_output = net(valset.x.to(device))
        val_loss[epoch] = criterion(val_output, valset.y.to(device))
        train_output = net(trainset.x.to(device))
        train_loss[epoch] = criterion(train_output, trainset.y.to(device))

        epoch += 1
        if pr and epoch % 500 == 0:
            print('epoch %7d val/train loss: %.2f/%.2f (%.2f%% for validation)\r' %
                  (epoch, val_loss[epoch - 1], train_loss[epoch - 1], val_loss[epoch - 1] / cfg['zero_val_loss'] * 100))
        if train_loss[epoch - 1] / cfg['zero_val_loss'] < cfg['stop_threshold_percent'] / 100:
            print('\rafter %d epochs, validation loss is %.2f, reached stop threshold of %.2f%% and training is done\r'
                  % (epoch, val_loss[epoch - 1], cfg['stop_threshold_percent']))
            converged = True
            stopping_criterion = 'converged'
            break
        training_time_so_far_in_minutes = (time.time() - total_training_time) / 60
        if training_time_so_far_in_minutes > cfg['max_training_time_in_minutes']:
            print('\rafter %d epochs, training time exceeded %d minutes threshold and training is stopped\r'
                  % (epoch, cfg['max_training_time_in_minutes']))
            stopping_criterion = 'time_out'
            break
        if np.isnan(epoch_loss[epoch - 1]):
            print('\rafter %d epochs, epoch loss is NaN and training is stopped\r'
                  % epoch)
            stopping_criterion = 'reached_nan'
            break
        if epoch == cfg['n_epochs']:
            print('\rreached maximal number of epochs %d and training is stopped\r'
                  % cfg['n_epochs'])
            stopping_criterion = 'epochs_over'
            break

    total_training_time = time.time() - total_training_time
    print(
        '\r\rfinished training! training time: %.2f minutes (optimization time: %.2f%%, iter time: %.2f%%, sampling time: %.2f%%\r\r)'
        % (total_training_time / 60, total_optimization_time / total_training_time * 100,
           total_iter_time / total_training_time * 100, total_sampling_time / total_training_time * 100))

    with tr.no_grad():
        val_output_after_training = net(valset.x.to(device))
        val_loss_after_training = criterion(val_output_after_training, valset.y.to(device))
        train_output_after_training = net(trainset.x.to(device))
        train_loss_after_training = criterion(train_output_after_training, trainset.y.to(device))
    print('\r\rfinal val/train loss is %.2f/%.2f\r\r (%.2f%% for validation)' % (
    val_loss_after_training, train_loss_after_training,
    val_loss_after_training / cfg['zero_val_loss'] * 100))

    return {'train_loss': train_loss[:epoch], 'val_loss': val_loss[:epoch], 'epoch_loss': epoch_loss[:epoch],
            'converged': converged, 'training_time': total_training_time,
            'stopping_criterion': stopping_criterion}


results = pd.DataFrame()

iters = 20
max_k = 30

sysid = int(sysstr)

f_output = open('output/test_' + testname + str(sysid) + '.txt', 'wt')
sys.stdout = f_output
f_resjson = 'results/test_' + testname + str(sysid) + '.json'
if sysid > iters * max_k:
    print('\rbad sys ID %d\r' % sysid)
else:
    k = (sysid - 1) // iters + 1
    print(k)
    i_iter = (sysid - 1) % iters

    ks = [k]
    cfg['ks'] = ks
    cfg['phases'] = rn.uniform(-np.pi, np.pi, len(cfg['ks']))
    print('\r\r*** ks =', ks, ', iter = %d\r' % i_iter)
    print('\rtest cfg: \r', cfg, '\r')

    total_loops_time = time.time()

    pr = False

    # generate new dataset
    datasets = gen_data(cfg, pr=False)
    cfg['zero_val_loss'] = tr.sum(datasets['trainset'].y ** 2).cpu().numpy() / len(datasets['trainset'])
    if pr: print('zero val loss is: %.3f\r' % cfg['zero_val_loss'])

    # create new network
    net = gen_net(cfg, device, use_parallel_gpus, pr)

    # optimize and save results
    my_dict = {'ks': ks, 'iter': i_iter, 'phases': cfg['phases']}
    my_dict.update(optimize(cfg, device, net, datasets, pr))
    results = results.append(my_dict, ignore_index=True)
    results.to_json(f_resjson)

    total_loops_time = (time.time() - total_loops_time) / 60
    print('\rfinished looping! total looping time: %.2f minutes\r' % total_loops_time)


f_output.close()
